library(CellChat)
library(BiocParallel)
library(MoleculeExperiment)
library(SingleCellExperiment)
library(SpatialExperiment)
library(scater)
library(SingleR)
library(scran)
library(future) 
library(future.apply) 
library(Matrix)
library(data.table) 
library(tidyverse) 
library(anndata)
library(biomaRt)

#' Compute Cell-to-Cell Communication scores
#' @param spe A SpatialExperiment object containing logcounts and spatialCoords.
#' @return A CellChat object with computed communication probabilities.
#' @details This function uses the human CellChatDB and truncatedMean for signaling inference. Any other database can also be used
run_cellchat <- function(spe){
  # Setup Parameters 
  conversion.factor = 1
  spot.size = 10 
  spatial.factors = data.frame(ratio = conversion.factor, tol = spot.size/2)
  # Data Preparation 
  data.input <- SingleCellExperiment::logcounts(spe)
  meta <- as.data.frame(SingleCellExperiment::colData(spe)) 
  meta$labels <- meta[["celltype"]]
  # Initialize CellChat with spatial coordinates
  cellchat <- CellChat::createCellChat(object = data.input, meta = meta, group.by = "celltype",
                             datatype = "spatial", coordinates = spatialCoords(spe), spatial.factors = spatial.factors)
  # Database Selection
  CellChatDB <- CellChatDB.human 
  cellChatDB.use <- CellChatDB
  cellchat@DB <- cellChatDB.use
  cellchat <- CellChat::subsetData(cellchat)
  # Signaling Inference
  cellchat <- CellChat::identifyOverExpressedGenes(cellchat, do.DE = FALSE, min.cells = 10)
  # Allocate memory for large matrices (250GB)
  options(future.globals.maxSize = 250 * 1024^3)
  cellchat <- CellChat::identifyOverExpressedInteractions(cellchat)
  # Compute the communication scores using a modified statistics
  cellchat <- computeCommunProbmod(cellchat, type = "truncatedMean", trim = 0.1,
                                    distance.use = TRUE, interaction.range = 100, scale.distance = 1,
                                    contact.dependent = TRUE, contact.range = 50, nboot = 20, population.size = TRUE)
  cellchat <- filterCommunication(cellchat, min.cells = 10)
  return(cellchat)
}



#' This is a core inference function that calculates the probability of cell-cell communication by integrating gene expression
#' and spatial distance.  It uses a permutation test to calculate p-values.
#' @param object A CellChat object with @data.signaling and @DB already populated.
#' @param type Method for calculating average gene expression per cluster. 
#'   Options: "triMean", "truncatedMean" (robust to outliers), "thresholdedMean", or "median".
#' @param trim The fraction (0 to 0.5) of observations to be trimmed from each end 
#'   of the distribution before the mean is computed. Used if type = "truncatedMean".
#' @param LR.use A data frame or vector of specific Ligand-Receptor pairs to test. 
#'   If NULL, the function uses all signaling interactions in object@LR$LRsig.
#' @param raw.use Logical. If TRUE, uses normalized count data. If FALSE, uses 
#'   imputed/projected data.
#' @param population.size Logical. If TRUE, scales the communication probability 
#'   by the number of cells in the source and target clusters (proportional to group size).
#' @param distance.use Logical. If TRUE, incorporates spatial coordinates to 
#'   penalize/constrain interactions based on physical distance between spots/cells.
#' @param interaction.length The maximum physical distance (in the same units as 
#'   spatialCoords) that a signal is expected to travel. 
#' @param scale.distance A numeric scaling factor for spatial coordinates. 
#'   Required to ensure the internal distance math remains stable (typically 0.1, 0.01, etc.).
#' @param k.min Minimum number of nearest neighbors used when calculating 
#'   inter-cluster distances in spatial mode.
#' @param nboot Number of permutations for the bootstrap test to calculate p-values. 
#'   Higher values (100+) increase precision but take more time.
#' @param seed.use Random seed for reproducibility of the permutation test.
#' @param Kh The parameter in the Hill function (half-maximal response) that 
#'   defines the expression level where signaling probability is 0.5.
#' @param n The Hill coefficient. Controls the sensitivity/steepness of the 
#'   probability curve in response to changes in gene expression.
#' @return A CellChat object with updated @net slot containing probs and pvals.
computeCommunProb_singlecell <- function(object, type = c("triMean", "truncatedMean","thresholdedMean", "median"), trim = 0.1, LR.use = NULL, raw.use = TRUE, population.size = FALSE,
                              distance.use = TRUE, interaction.length = 200, scale.distance = 0.01, k.min = 10,
                              nboot = 100, seed.use = 1L, Kh = 0.5, n = 1) {
  type <- match.arg(type)
  cat(type, "is used for calculating the average gene expression per cell group.", "\n")
  FunMean <- switch(type,
                    triMean = triMean,
                    truncatedMean = function(x) mean(x, trim = trim, na.rm = TRUE),
                    median = function(x) median(x, na.rm = TRUE))
  
  if (raw.use) {
    data <- as.matrix(object@data.signaling)
  } else {
    data <- object@data.project
  }
  
  if (is.null(LR.use)) {
    pairLR.use <- object@LR$LRsig
  } else {
    pairLR.use <- LR.use
  }
  
  complex_input <- object@DB$complex
  cofactor_input <- object@DB$cofactor
  
  my.sapply <- ifelse(
    test = future::nbrOfWorkers() == 1,
    yes = sapply,
    no = future.apply::future_sapply
  )
  
  pairLRsig <- pairLR.use
  group <- object@idents
  geneL <- as.character(pairLRsig$ligand)
  geneR <- as.character(pairLRsig$receptor)
  nLR <- nrow(pairLRsig)
  numCluster <- nlevels(group)
  data.use <- data/max(data)
  nC <- ncol(data.use)
  
  # compute the average expression per group
  data.use.avg <- aggregate(t(data.use), list(group), FUN = FunMean)
  data.use.avg <- t(data.use.avg[,-1])
  colnames(data.use.avg) <- levels(group)
  # compute the expression of ligand or receptor
  dataLavg <- CellChat::computeExpr_LR(geneL, data.use.avg, complex_input)
  dataRavg <- CellChat::computeExpr_LR(geneR, data.use.avg, complex_input)
  # take account into the effect of co-activation and co-inhibition receptors
  dataRavg.co.A.receptor <- CellChat::computeExpr_coreceptor(cofactor_input, data.use.avg, pairLRsig, type = "A")
  dataRavg.co.I.receptor <- CellChat::computeExpr_coreceptor(cofactor_input, data.use.avg, pairLRsig, type = "I")
  dataRavg <- dataRavg * dataRavg.co.A.receptor/dataRavg.co.I.receptor
  
  
  # compute the spatial constraint
  if (object@options$datatype != "RNA") {
    data.spatial <- object@images$coordinates
    spot.size.fullres <- object@images$scale.factors$spot
    spot.size <- object@images$scale.factors$spot.diameter
    d.spatial <- CellChat::computeRegionDistance(coordinates = data.spatial, group = group, trim = trim, interaction.length = interaction.length, spot.size = spot.size, spot.size.fullres = spot.size.fullres, k.min = k.min)
    
    if (distance.use) {
      print(paste0('>>> Run CellChat on spatial imaging data using distances as constraints <<< [', Sys.time(),']'))
      d.spatial <- d.spatial * scale.distance
      diag(d.spatial) <- NaN
      cat("The suggested minimum value of scaled distances is in [1,2], and the calculated value here is ", min(d.spatial, na.rm = TRUE),"\n")
      if (min(d.spatial, na.rm = TRUE) < 1) {
        stop("Please increase the value of `scale.distance` and check the suggested values in the parameter description (e.g., 1, 0.1, 0.01, 0.001, 0.11, 0.011)")
      }
      P.spatial <- 1/d.spatial
      P.spatial[is.na(d.spatial)] <- 0
      diag(P.spatial) <- max(P.spatial) 
      d.spatial <- d.spatial/scale.distance
    } else {
      print(paste0('>>> Run CellChat on spatial imaging data without distances as constraints <<< [', Sys.time(),']'))
      P.spatial <- matrix(1, nrow = numCluster, ncol = numCluster)
      P.spatial[is.na(d.spatial)] <- 0
    }
    
  } else {
    print(paste0('>>> Run CellChat on sc/snRNA-seq data <<< [', Sys.time(),']'))
    d.spatial <- matrix(NaN, nrow = numCluster, ncol = numCluster)
    P.spatial <- matrix(1, nrow = numCluster, ncol = numCluster)
    distance.use = NULL; interaction.length = NULL; spot.size = NULL; spot.size.fullres = NULL; k.min = NULL;
  }
  
  Prob <- array(0, dim = c(numCluster,numCluster,nLR))
  Pval <- array(0, dim = c(numCluster,numCluster,nLR))
  P1 <- array(0, dim = c(numCluster,numCluster,nLR))
  
  set.seed(seed.use)
  permutation <- replicate(nboot, sample.int(nC, size = nC))
  data.use.avg.boot <- my.sapply(
    X = 1:nboot,
    FUN = function(nE) {
      groupboot <- group[permutation[, nE]]
      data.use.avgB <- aggregate(t(data.use), list(groupboot), FUN = FunMean)
      data.use.avgB <- t(data.use.avgB[,-1])
      return(data.use.avgB)
    },
    simplify = FALSE
  )
  pb <- txtProgressBar(min = 0, max = nLR, style = 3, file = stderr())
  
  for (i in 1:nLR) {
    # ligand/receptor
    dataLR <- Matrix::crossprod(matrix(dataLavg[i,], nrow = 1), matrix(dataRavg[i,], nrow = 1))
    P1 <- dataLR^n/(Kh^n + dataLR^n)
    P1_Pspatial <- P1*P.spatial
    if (sum(P1_Pspatial) == 0) {
      Pnull = P1_Pspatial
      Prob[ , , i] <- Pnull
      p = 1
      Pval[, , i] <- matrix(p, nrow = numCluster, ncol = numCluster, byrow = FALSE)
    } else {
      
      Pnull = P1*P.spatial 
      Prob[ , , i] <- Pnull
      P1[ , , i] <- P1
      Pnull <- as.vector(Pnull)
    
      Pboot <- sapply(
        X = 1:nboot,
        FUN = function(nE) {
          data.use.avgB <- data.use.avg.boot[[nE]]
          dataLavgB <- computeExpr_LR(geneL[i], data.use.avgB, complex_input)
          dataRavgB <- computeExpr_LR(geneR[i], data.use.avgB, complex_input)
          # take account into the effect of co-activation and co-inhibition receptors
          dataRavgB.co.A.receptor <- computeExpr_coreceptor(cofactor_input, data.use.avgB, pairLRsig[i, , drop = FALSE], type = "A")
          dataRavgB.co.I.receptor <- computeExpr_coreceptor(cofactor_input, data.use.avgB, pairLRsig[i, , drop = FALSE], type = "I")
          dataRavgB <- dataRavgB * dataRavgB.co.A.receptor/dataRavgB.co.I.receptor
          dataLRB = Matrix::crossprod(dataLavgB, dataRavgB)
          P1.boot <- dataLRB^n/(Kh^n + dataLRB^n)
          Pboot = P1.boot*P.spatial
          return(as.vector(Pboot))
        }
      )
      
      Pboot <- matrix(unlist(Pboot), nrow=length(Pnull), ncol = nboot, byrow = FALSE)
      nReject <- rowSums(Pboot - Pnull > 0)
      p = nReject/nboot
      Pval[, , i] <- matrix(p, nrow = numCluster, ncol = numCluster, byrow = FALSE)
    }
    setTxtProgressBar(pb = pb, value = i)
  }
  close(con = pb)
  Pval[Prob == 0] <- 1
  dimnames(Prob) <- list(levels(group), levels(group), rownames(pairLRsig))
  dimnames(Pval) <- dimnames(Prob)
  net <- list("prob" = Prob, "pval" = Pval, "P1" = P1)
  object@options$run.time <- as.numeric(execution.time, units = "secs")
  
  object@options$parameter <- list(type.mean = type, trim = trim, raw.use = raw.use, population.size = population.size,  nboot = nboot, seed.use = seed.use, Kh = Kh, n = n,
                                   distance.use = distance.use, interaction.length = interaction.length, spot.size = spot.size, spot.size.fullres = spot.size.fullres, k.min = k.min
  )
  if (object@options$datatype != "RNA") {
    object@images$distance <- d.spatial
  }
  object@net <- net
  print(paste0('>>> CellChat inference is done. Parameter values are stored in `object@options$parameter` <<<'))
  return(object)
}

#' Process Xenium data and infer cell–cell communication. Generates cell, cytoplasmic, and nuclear SpatialExperiment objects from
#' 10x Xenium data, performs QC, normalization, cell type annotation with SingleR, and calculates communication score on each compartment.
#' @param input_filepath Character. Path to the Xenium output directory.
#' @param genes_file List. Xenium gene metadata used to identify negative control probes.
#' @param ref SingleCellExperiment. Reference dataset with cell type labels in for SingleR-based annotation
#' @param log Boolean, if logcounts are existing in the assay of the reference
#' @return Named list of objects for cell cytoplasm and nucleus regions
process_st_data <- function(input_filepath, genes_file, ref, log = T){
  # read into molecule experiment object
  me = read_Xenium(input_filepath, keepCols = "essential", addBoundaries = c("cell", "nucleus"))

  # compartment separation
  spe_cell <- countMolecules(
    me, boundariesAssay = "cell", nCores = 34)
  spe_nuc <- countMolecules(
    me, boundariesAssay = "nucleus", nCores = 34)
  spe_cyt = spe_cell[, spe_cell@colData$cell_id %in% spe_nuc@colData$cell_id]
  spe_cyt@assays@data@listData[["counts"]] = spe_cyt@assays@data@listData[["counts"]] - spe_nuc@assays@data@listData[["counts"]] 
  # since the cytoplasm might have -ve values, removing those cells in all the spe objects
  mat = spe_cyt@assays@data@listData[["counts"]]%>%as.matrix()
  filter_cells <- colSums(mat < 0) > 0
  # remove the cells
  spe_cyt <- spe_cyt[, !filter_cells]
  spe_cell = spe_cell[, spe_cell@colData$cell_id %in% spe_cyt@colData$cell_id]
  spe_nuc = spe_nuc[, spe_nuc@colData$cell_id %in% spe_cyt@colData$cell_id]
  spe_list = list(cell = spe_cell, cyt = spe_cyt, nuc = spe_nuc)
  
  # remove negative control probes and low quality cells
  genes = genes_file[["payload"]][["targets"]][["type"]]%>%as.data.table()
  thresh_sum = 20
  thresh_detected = 10
  spe_list = lapply(spe_list, function(spe){
    is_neg = c(unlist(genes[descriptor != "gene",]$data.name), rownames(spe)[grepl('*Codeword*',rownames(spe))])
    spe = addPerCellQCMetrics(spe, subsets = list("neg" = is_neg))
    spe$keep = spe$sum > thresh_sum & spe$detected > thresh_detected
    spe = spe[, spe$keep]
    spe = spe[!(rownames(spe) %in% is_neg), ]
    return(spe)
  })
  
  # Normalization
  spe_list = lapply(spe_list, function(spe){
    spe = logNormCounts(spe)
    return(spe)
  })
  
  spe = spe_list$cell
  
  # Annotation
  if (log == T){
  ref = logNormCounts(ref)
  hvg <- modelGeneVar(ref)
  top_genes <- getTopHVGs(hvg, n = 2000)
  ref_sub <- ref[top_genes, ] # subset reference to top genes for speed
  }
  else {
    # pass
  }
  prediction = SingleR(test = spe, ref = ref_sub, labels = ref$cell_type)
  spe$celltype = prediction$labels
  # annotate the same cell_type in cytoplasm and nucleus
  map = colData(spe)[,c("celltype", "cell_id")]
  spe_cyt = spe_list$cyt
  spe_nuc = spe_list$nuc
  spe_cell = spe
  spe_cyt$celltype = map$celltype[match(spe_cyt$cell_id, map$cell_id)]
  spe_nuc$celltype = map$celltype[match(spe_nuc$cell_id, map$cell_id)]
  spe_list = list(spe_cell, spe_cyt, spe_nuc)
  names(spe_list) = c("cell", "cyt", "nuc")
  
  # communication scores
  future::plan("multisession", workers = 4)
  obj = lapply(spe_list, run_cellchat)
  return (obj)
}

# gene id to symbol map
geneid_to_symbol <- function(){
  ensembl <- useEnsembl(biomart = "genes", dataset = "hsapiens_gene_ensembl")
  map = getBM(attributes = c('ensembl_gene_id', 'hgnc_symbol'),
              mart = ensembl) 
  return(map)
}

mapping_function <- function(ref_dataset){
  map = geneid_to_symbol()
  mapped_symbols <- map$hgnc_symbol[match(rownames(ref_dataset), map$ensembl_gene_id)]
  mapped_symbols <- ifelse(is.na(mapped_symbols),rownames(ref_dataset),mapped_symbols)
  newrownames <- make.unique(mapped_symbols)
  rownames(ref_dataset) <- new_rownames
  names(assays(ref_dataset))[1] = "counts"
  ref_dataset$ident = ref_dataset$cell_type
  return(ref_dataset)
}
