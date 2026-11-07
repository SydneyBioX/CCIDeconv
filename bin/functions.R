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

library(CellChat)
library(BiocParallel)

#' Compute Cell-to-Cell Communication scores
#' @param spe A SpatialExperiment object containing logcounts and spatialCoords.
#' @return A CellChat object with computed communication probabilities.
#' @details This function uses the human CellChatDB and truncatedMean for signaling inference. Any other database can also be used
run_cellchat <- function(spe){
  conversion.factor = 1
  spot.size = 10 
  spatial.factors = data.frame(ratio = conversion.factor, tol = spot.size/2)
  data.input <- SingleCellExperiment::logcounts(spe)
  meta <- as.data.frame(SingleCellExperiment::colData(spe)) # extract a dataframe of the cell labels
  meta$labels <- meta[["celltype"]]
  cellchat <- createCellChat(object = data.input, meta = meta, group.by = "celltype",
                             datatype = "spatial", coordinates = spatialCoords(spe), spatial.factors = spatial.factors)
  CellChatDB <- CellChatDB.human 
  cellChatDB.use <- CellChatDB
  cellchat@DB <- cellChatDB.use
  cellchat <- subsetData(cellchat)
  cellchat <- identifyOverExpressedGenes(cellchat, do.DE = FALSE, min.cells = 10)
  # options(future.globals.maxSize = 250 * 1024^3)
  cellchat <- identifyOverExpressedInteractions(cellchat)
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
computeCommunProbmod <- function(object, type = c("triMean", "truncatedMean","thresholdedMean", "median"), trim = 0.1, LR.use = NULL, raw.use = TRUE, population.size = FALSE,
                                 distance.use = TRUE, interaction.range = 250, scale.distance = 0.01, k.min = 10, contact.dependent = TRUE, contact.range = NULL, contact.knn.k = NULL, 
                                 contact.dependent.forced = FALSE, do.symmetric = TRUE, nboot = 100, seed.use = 1L, Kh = 0.5, n = 1) {
  type <- match.arg(type)
  cat(type, "is used for calculating the average gene expression per cell group.", "\n")
  FunMean <- switch(type,
                    triMean = triMean,
                    truncatedMean = function(x) mean(x, trim = trim, na.rm = TRUE),
                    thresholdedMean = function(x) thresholdedMean(x, trim = trim, na.rm = TRUE),
                    median = function(x) median(x, na.rm = TRUE))
  
  if (raw.use) {
    data <- as.matrix(object@data.signaling)
  } else {
    data <- as.matrix(object@data.smooth)
  }
  if (is.null(LR.use)) {
    pairLR.use <- object@LR$LRsig
  } else {
    if (length(unique(LR.use$annotation)) > 1) {
      LR.use$annotation <- factor(LR.use$annotation, levels = c("Secreted Signaling","ECM-Receptor", "Non-protein Signaling", "Cell-Cell Contact"))
      LR.use <- LR.use[order(LR.use$annotation), , drop = FALSE]
      LR.use$annotation <- as.character(LR.use$annotation)
    }
    pairLR.use <- LR.use
  }
  complex_input <- object@DB$complex
  cofactor_input <- object@DB$cofactor
  if (future::nbrOfWorkers() == 1) {
    my.sapply <- sapply
  } else {
    my.sapply <- future.apply::future_sapply
  }
  
  ptm = Sys.time()
  
  pairLRsig <- pairLR.use
  group <- object@idents
  geneL <- as.character(pairLRsig$ligand)
  geneR <- as.character(pairLRsig$receptor)
  nLR <- nrow(pairLRsig)
  numCluster <- nlevels(group)
  if (numCluster != length(unique(group))) {
    stop("Please check `unique(object@idents)` and ensure that the factor levels are correct!
         You may need to drop unused levels using 'droplevels' function. e.g.,
         `meta$labels = droplevels(meta$labels, exclude = setdiff(levels(meta$labels),unique(meta$labels)))`")
  }
  
  data.use <- data/max(data)
  nC <- ncol(data.use)
  
  # compute the average expression per group
  data.use.avg <- aggregate(t(data.use), list(group), FUN = FunMean)
  data.use.avg <- t(data.use.avg[,-1])
  colnames(data.use.avg) <- levels(group)
  # compute the expression of ligand or receptor
  dataLavg <- computeExpr_LR(geneL, data.use.avg, complex_input)
  dataRavg <- computeExpr_LR(geneR, data.use.avg, complex_input)
  # take account into the effect of co-activation and co-inhibition receptors
  dataRavg.co.A.receptor <- computeExpr_coreceptor(cofactor_input, data.use.avg, pairLRsig, type = "A")
  dataRavg.co.I.receptor <- computeExpr_coreceptor(cofactor_input, data.use.avg, pairLRsig, type = "I")
  dataRavg <- dataRavg * dataRavg.co.A.receptor/dataRavg.co.I.receptor
  
  dataLavg2 <- t(replicate(nrow(dataLavg), as.numeric(table(group))/nC))
  dataRavg2 <- dataLavg2
  
  # compute the expression of agonist and antagonist
  index.agonist <- which(!is.na(pairLRsig$agonist) & pairLRsig$agonist != "")
  index.antagonist <- which(!is.na(pairLRsig$antagonist) & pairLRsig$antagonist != "")
  # quantify the communication probability
  
  # compute the spatial constraint
  if (object@options$datatype != "RNA") {
    data.spatial <- object@images$coordinates
    if ("spatial.factors" %in% names(object@images)) {
      ratio <- object@images$spatial.factors$ratio
      tol <- object@images$spatial.factors$tol
    } else {
      stop("`object@images$spatial.factors` is missing. Please update the object via `updateCellChat`! \n")
    }
    
    meta.t = data.frame(group = group, samples = object@meta$samples, row.names = rownames(object@meta))
    res <- computeRegionDistance(coordinates = data.spatial, meta = meta.t, interaction.range = interaction.range, ratio = ratio, tol = tol, k.min = k.min, contact.dependent = contact.dependent, contact.range = contact.range, contact.knn.k = contact.knn.k)
    d.spatial <- res$d.spatial # NaN if no nearby cell pairs
    adj.contact <- res$adj.contact # zeros if no nearby cell pairs
    if (distance.use) {
      print(paste0('>>> Run CellChat on spatial transcriptomics data using distances as constraints of the computed communication probability <<< [', Sys.time(),']'))
      d.spatial <- d.spatial * scale.distance
      diag(d.spatial) <- NaN
      d.min <- min(d.spatial, na.rm = TRUE)
      if (d.min < 1) {
        cat("The suggested minimum value of scaled distances is in [1,2], and the calculated value here is ", d.min,"\n")
        stop("Please increase the value of `scale.distance` and use a value that is slighly smaller than ", format(1/d.min, digits = 2) ,"\n")
      }
      P.spatial <- 1/d.spatial
      P.spatial[is.na(d.spatial)] <- 0
      diag(P.spatial) <- max(P.spatial) # if this value is 1, the self-connections will have more larger weight.
      d.spatial <- d.spatial/scale.distance # This is only for saving the data
    } else {
      print(paste0('>>> Run CellChat on spatial transcriptomics data without distance values as constraints of the computed communication probability <<< [', Sys.time(),']'))
      P.spatial <- matrix(1, nrow = numCluster, ncol = numCluster)
      P.spatial[is.na(d.spatial)] <- 0 # diagonal is 1
    }
    
  } else {
    print(paste0('>>> Run CellChat on sc/snRNA-seq data <<< [', Sys.time(),']'))
    d.spatial <- matrix(NaN, nrow = numCluster, ncol = numCluster)
    P.spatial <- matrix(1, nrow = numCluster, ncol = numCluster)
    adj.contact <- matrix(1, nrow = numCluster, ncol = numCluster)
    contact.dependent = FALSE; contact.dependent.forced = FALSE; contact.range = NULL; contact.knn.k = NULL;
    distance.use = NULL; interaction.range = NULL; ratio = NULL; tol = NULL; k.min = NULL;
  }
  
  if (object@options$datatype == "RNA") {
    nLR1 <- nLR
  } else {
    if (contact.dependent.forced == TRUE) {
      cat("Force to run CellChat in a `contact-dependent` manner for all L-R pairs including secreted signaling.\n")
      P.spatial <- P.spatial * adj.contact
      nLR1 <- nLR
    } else { # contact.dependent.forced == F
      if (contact.dependent == TRUE && length(unique(pairLRsig$annotation)) > 0) {
        if (all(unique(pairLRsig$annotation) %in% c("Cell-Cell Contact"))) {
          cat("All the input L-R pairs are `Cell-Cell Contact` signaling. Run CellChat in a contact-dependent manner. \n")
          P.spatial <- P.spatial * adj.contact
          nLR1 <- nLR
        } else if (all(unique(pairLRsig$annotation) %in% c("Secreted Signaling", "ECM-Receptor", "Non-protein Signaling"))) {
          cat("Molecules of the input L-R pairs are diffusible. Run CellChat in a diffusion manner based on the `interaction.range`.\n")
          nLR1 <- nLR
        } else {
          cat("The input L-R pairs have both secreted signaling and contact-dependent signaling. Run CellChat in a contact-dependent manner for `Cell-Cell Contact` signaling, and in a diffusion manner based on the `interaction.range` for other L-R pairs. \n")
          nLR1 <- max(which(pairLRsig$annotation %in% c("Secreted Signaling", "ECM-Receptor", "Non-protein Signaling")))
        }
      } else { # contact.dependent == F or there is no `annotation` column in the database
        cat("Run CellChat in a diffusion manner based on the `interaction.range` for all L-R pairs. Setting `contact.dependent = TRUE` if preferring a contact-dependent manner for `Cell-Cell Contact` signaling. \n")
        nLR1 <- nLR
      }
    }
  }
  
  
  #**here**
  Prob <- array(0, dim = c(numCluster,numCluster,nLR))
  # Prob_P1 <- array(0, dim = c(numCluster,numCluster,nLR))
  # Prob_P2 <- array(0, dim = c(numCluster,numCluster,nLR))
  # Prob_P3 <- array(0, dim = c(numCluster,numCluster,nLR))
  # Prob_P4 <- array(0, dim = c(numCluster,numCluster,nLR))
  # Prob_P.spatial <- array(0, dim = c(numCluster,numCluster,nLR))
  Prob_P2_P3_P4 <- array(0, dim = c(numCluster,numCluster,nLR))
  
  # list_P1 <- array(0, dim = c(numCluster,numCluster,nLR))
  # list_P2 <- array(0, dim = c(numCluster,numCluster,nLR))
  # list_P3 <- array(0, dim = c(numCluster,numCluster,nLR))
  # list_P4 <- array(0, dim = c(numCluster,numCluster,nLR))
  # list_P.spatial <- array(0, dim = c(numCluster,numCluster,nLR))
  
  Pval <- array(0, dim = c(numCluster,numCluster,nLR))
  # PvalP1 <- array(0, dim = c(numCluster,numCluster,nLR))
  # PvalP2 <- array(0, dim = c(numCluster,numCluster,nLR))
  # PvalP3 <- array(0, dim = c(numCluster,numCluster,nLR))
  # PvalP4 <- array(0, dim = c(numCluster,numCluster,nLR))
  # PvalP.spatial <- array(0, dim = c(numCluster,numCluster,nLR))
  PvalP2_P3_P4 <- array(0, dim = c(numCluster,numCluster,nLR))
  
  set.seed(seed.use)
  # reshuffling is happenign here
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
  
  # here is the calculation of the actual data under the null hypotheiss
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
      if (i > nLR1) {
        P.spatial <- P.spatial * adj.contact
      }
      # agonist and antagonist
      if (is.element(i, index.agonist)) {
        data.agonist <- computeExpr_agonist(data.use = data.use.avg, pairLRsig, cofactor_input, index.agonist = i, Kh = Kh,  n = n)
        P2 <- Matrix::crossprod(matrix(data.agonist, nrow = 1))
      } else {
        P2 <- matrix(1, nrow = numCluster, ncol = numCluster)
      }
      if (is.element(i, index.antagonist)) {
        data.antagonist <- computeExpr_antagonist(data.use = data.use.avg, pairLRsig, cofactor_input,  index.antagonist = i, Kh = Kh,  n = n)
        P3 <- Matrix::crossprod(matrix(data.antagonist, nrow = 1))
      } else {
        P3 <- matrix(1, nrow = numCluster, ncol = numCluster)
      }
      # number of cells
      if (population.size) {
        P4 <- Matrix::crossprod(matrix(dataLavg2[i,], nrow = 1), matrix(dataRavg2[i,], nrow = 1))
      } else {
        P4 <- matrix(1, nrow = numCluster, ncol = numCluster)
      }
      
      #  interaction scores
      Pnull = P1*P2*P3*P4*P.spatial
      # P1_ablated = P2*P3*P4*P.spatial
      # P2_ablated = P1*P3*P4*P.spatial
      # P3_ablated = P1*P2*P4*P.spatial
      # P4_ablated = P1*P2*P3*P.spatial
      # P.spatial_ablated = P1*P2*P3*P4
      P2_P3_P4_ablated = P1*P.spatial
      
      # return the matrix of interaction scores
      Prob[ , , i] <- Pnull
      # Prob_P1[ , , i] <- P1_ablated
      # Prob_P2[ , , i] <- P2_ablated
      # Prob_P3[ , , i] <- P3_ablated
      # Prob_P4[ , , i] <- P4_ablated
      # Prob_P.spatial[ , , i] <- P.spatial_ablated
      Prob_P2_P3_P4[ , , i] <- P2_P3_P4_ablated
      
      # store the values as an object, # return the individual values
      # list_P1[ , , i] <- P1
      # list_P2[ , , i] <- P2
      # list_P3[ , , i] <- P3
      # list_P4[ , , i] <- P4
      # list_P.spatial[ , , i] <- P.spatial
      
      Pnull <- as.vector(Pnull)
      # P1_ablated <- as.vector(P1_ablated)
      # P2_ablated <- as.vector(P2_ablated)
      # P3_ablated <- as.vector(P3_ablated)
      # P4_ablated <- as.vector(P4_ablated)
      # P.spatial_ablated <- as.vector(P.spatial_ablated)
      P2_P3_P4_ablated <- as.vector(P2_P3_P4_ablated)
      
      # here is the calculation of each permutation
      # Pboot <- foreach(nE = 1:nboot) %dopar% {
      Pboot_ablated <- sapply(
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
          # agonist and antagonist
          if (is.element(i, index.agonist)) {
            data.agonist <- computeExpr_agonist(data.use = data.use.avgB, pairLRsig, cofactor_input, index.agonist = i, Kh = Kh,  n = n)
            P2.boot <- Matrix::crossprod(matrix(data.agonist, nrow = 1))
          } else {
            P2.boot <- matrix(1, nrow = numCluster, ncol = numCluster)
          }
          if (is.element(i, index.antagonist)) {
            data.antagonist <- computeExpr_antagonist(data.use = data.use.avgB, pairLRsig, cofactor_input, index.antagonist = i, Kh = Kh,  n= n)
            P3.boot <- Matrix::crossprod(matrix(data.antagonist, nrow = 1))
          } else {
            P3.boot <- matrix(1, nrow = numCluster, ncol = numCluster)
          }
          
          if (population.size) {
            groupboot <- group[permutation[, nE]]
            dataLavg2B <- as.numeric(table(groupboot))/nC
            dataLavg2B <- matrix(dataLavg2B, nrow = 1)
            dataRavg2B <- dataLavg2B
            P4.boot = Matrix::crossprod(dataLavg2B, dataRavg2B)
          } else {
            P4.boot = matrix(1, nrow = numCluster, ncol = numCluster)
          }
          
          #  Pboot = P1.boot*P2.boot*P3.boot*P4.boot
          Pboot = as.vector(P1.boot*P2.boot*P3.boot*P4.boot*P.spatial)
          # PbootP1 = as.vector(P2.boot*P3.boot*P4.boot*P.spatial)
          # PbootP2 = as.vector(P1.boot*P3.boot*P4.boot*P.spatial)
          # PbootP3 = as.vector(P1.boot*P2.boot*P4.boot*P.spatial)
          # PbootP4 = as.vector(P1.boot*P2.boot*P3.boot*P.spatial)
          # PbootP.spatial = as.vector(P1.boot*P2.boot*P3.boot*P4.boot)
          PbootP2_P3_P4 = as.vector(P1.boot*P.spatial)
          
          ablatedboot = list('Pboot' = Pboot, "PbootP2_P3_P4"= PbootP2_P3_P4)
          return(ablatedboot)
        }
      )
      
      # Full model
      Pboot = Pboot_ablated[1,]
      Pboot <- matrix(unlist(Pboot), nrow=length(Pnull), ncol = nboot, byrow = FALSE)
      nReject <- rowSums(Pboot - Pnull > 0)
      p = nReject/nboot
      Pval[, , i] <- matrix(p, nrow = numCluster, ncol = numCluster, byrow = FALSE)
      
      # ablated models
      # P1
      # Pboot = Pboot_ablated[2,]
      # Pboot <- matrix(unlist(Pboot), nrow=length(P1_ablated), ncol = nboot, byrow = FALSE)
      # nReject <- rowSums(Pboot - P1_ablated > 0)
      # p = nReject/nboot
      # PvalP1[, , i] <- matrix(p, nrow = numCluster, ncol = numCluster, byrow = FALSE)
      # 
      # # P2
      # Pboot = Pboot_ablated[3,]
      # Pboot <- matrix(unlist(Pboot), nrow=length(P2_ablated), ncol = nboot, byrow = FALSE)
      # nReject <- rowSums(Pboot - P2_ablated > 0)
      # p = nReject/nboot
      # PvalP2[, , i] <- matrix(p, nrow = numCluster, ncol = numCluster, byrow = FALSE)
      # 
      # # P3
      # Pboot = Pboot_ablated[4,]
      # Pboot <- matrix(unlist(Pboot), nrow=length(P3_ablated), ncol = nboot, byrow = FALSE)
      # nReject <- rowSums(Pboot - P3_ablated > 0)
      # p = nReject/nboot
      # PvalP3[, , i] <- matrix(p, nrow = numCluster, ncol = numCluster, byrow = FALSE)
      # 
      # # P4
      # Pboot = Pboot_ablated[5,]
      # Pboot <- matrix(unlist(Pboot), nrow=length(P4_ablated), ncol = nboot, byrow = FALSE)
      # nReject <- rowSums(Pboot - P4_ablated > 0)
      # p = nReject/nboot
      # PvalP4[, , i] <- matrix(p, nrow = numCluster, ncol = numCluster, byrow = FALSE)
      # 
      # # P.spatial
      # Pboot = Pboot_ablated[6,]
      # Pboot <- matrix(unlist(Pboot), nrow=length(P.spatial_ablated), ncol = nboot, byrow = FALSE)
      # nReject <- rowSums(Pboot - P.spatial_ablated > 0)
      # p = nReject/nboot
      # PvalP.spatial[, , i] <- matrix(p, nrow = numCluster, ncol = numCluster, byrow = FALSE)
      
      # P2 P3 and P4 
      Pboot = Pboot_ablated[2,]
      Pboot <- matrix(unlist(Pboot), nrow=length(P2_P3_P4_ablated), ncol = nboot, byrow = FALSE)
      nReject <- rowSums(Pboot - P2_P3_P4_ablated > 0)
      p = nReject/nboot
      PvalP2_P3_P4[, , i] <- matrix(p, nrow = numCluster, ncol = numCluster, byrow = FALSE)
      
    }
    setTxtProgressBar(pb = pb, value = i)
  }
  close(con = pb)
  
  # return p values
  Pval[Prob == 0] <- 1
  # PvalP1[Prob_P1 == 0] <- 1
  # PvalP2[Prob_P2 == 0] <- 1
  # PvalP3[Prob_P3 == 0] <- 1
  # PvalP4[Prob_P4 == 0] <- 1
  # PvalP.spatial[Prob_P.spatial == 0] <- 1
  PvalP2_P3_P4[Prob_P2_P3_P4 == 0] <- 1
  
  dimnames(Prob) <- list(levels(group), levels(group), rownames(pairLRsig))
  dimnames(Pval) <- dimnames(Prob) 
  
  # dimnames(Prob_P1) <- dimnames(Prob) 
  # dimnames(PvalP1) <- dimnames(Prob) 
  # dimnames(list_P1) <- dimnames(Prob) 
  # 
  # dimnames(Prob_P2) <- dimnames(Prob) 
  # dimnames(PvalP2) <- dimnames(Prob) 
  # dimnames(list_P2) <- dimnames(Prob) 
  # 
  # dimnames(Prob_P3) <- dimnames(Prob) 
  # dimnames(PvalP3) <- dimnames(Prob) 
  # dimnames(list_P3) <- dimnames(Prob) 
  # 
  # dimnames(Prob_P4) <- dimnames(Prob) 
  # dimnames(PvalP4) <- dimnames(Prob) 
  # dimnames(list_P4) <- dimnames(Prob) 
  # 
  # dimnames(Prob_P.spatial) <- dimnames(Prob) 
  # dimnames(PvalP.spatial) <- dimnames(Prob) 
  # dimnames(list_P.spatial) <- dimnames(Prob) 
  # 
  dimnames(Prob_P2_P3_P4) <- dimnames(Prob) 
  dimnames(PvalP2_P3_P4) <- dimnames(Prob) 
  
  net <- list("prob" = Prob, # 'P1_ablated' = Prob_P1, 'P2_ablated' = Prob_P2, 'P3_ablated' = Prob_P3, 'P4_ablated' = Prob_P4, 'P.spatial_ablated' = Prob_P.spatial, 
              "Prob_P2_P3_P4" = Prob_P2_P3_P4,
              "Pval" = Pval, # "PvalP1" = PvalP1, "PvalP2" = PvalP2, "PvalP3" = PvalP3, "PvalP4" = PvalP4, "PvalP.spatial" = PvalP.spatial, 
              "PvalP2_P3_P4" = PvalP2_P3_P4
              # "P1_values" = list_P1, "P2_values" = list_P2, "P3_values" = list_P3, "P4_values" = list_P4, "P.spatial_values" = list_P.spatial
  )
  execution.time = Sys.time() - ptm
  object@options$run.time <- as.numeric(execution.time, units = "secs")
  
  object@options$parameter <- list(type.mean = type, trim = trim, raw.use = raw.use, population.size = population.size,  nboot = nboot, seed.use = seed.use, Kh = Kh, n = n,
                                   distance.use = distance.use, interaction.range = interaction.range, ratio = ratio, tol = tol, k.min = k.min,
                                   contact.dependent = contact.dependent, contact.range = contact.range, contact.knn.k = contact.knn.k, contact.dependent.forced = contact.dependent.forced
  )
  if (object@options$datatype != "RNA") {
    object@images$distance <- d.spatial
  }
  object@net <- net
  print(paste0('>>> CellChat inference is done. Parameter values are stored in `object@options$parameter` <<< [', Sys.time(),']'))
  return(object)
}

#' Process Xenium data and infer cell–cell communication. Generates cell, cytoplasmic, and nuclear SpatialExperiment objects from
#' 10x Xenium data, performs QC, normalization, cell type annotation with SingleR, and calculates communication score on each compartment.
#' @param input_filepath Character. Path to the Xenium output directory.
#' @param genes_file List. Xenium gene metadata used to identify negative control probes.
#' @param ref SingleCellExperiment. Reference dataset with cell type labels in for SingleR-based annotation
#' @param log Boolean, if logcounts are existing in the assay of the reference
#' @param cellchat Boolean, if set to True only calculates the score, else outputs the formula parts
#' @return Named list of objects for cell cytoplasm and nucleus regions
process_st_data <- function(input_filepath, genes_file, ref, log = T, cellchat = TRUE){
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
  if (cellchat == TRUE){
  obj = lapply(spe_list, run_cellchat)
  } else {
  obj = lapply(spe_list, run_cellchat_formula)
  }
  return (obj)
}

# gene id to symbol map
#' @return give a mapping between gene id and gene symbol
geneid_to_symbol <- function(){
  ensembl <- useEnsembl(biomart = "genes", dataset = "hsapiens_gene_ensembl")
  map = getBM(attributes = c('ensembl_gene_id', 'hgnc_symbol'),
              mart = ensembl) 
  return(map)
}

#' Function to add hgnc symbol to rownames of reference single cell experiment 
#'@param ref_dataset single cell experiment reference to be mapped
#' @return retruns an single cell experiment object with the symbols as rownames
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
