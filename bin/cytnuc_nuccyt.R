# This is the function to check for LR interaction between the cytoplasm and nucleus occurs 
# load the cell chat object of interest
library(CellChat)
library(future)


#' Calculate communication scores between cytoplasm to nucleus or nucleus to cytoplasm
#' This function computes ligand-receptor communication scores between two
#' CellChat objects, representing either a cytoplasm or a nucleus. The communication
#' direction is determined by which object is assigned as the ligand and which
#' as the receptor.
#' The function calculates communication scores for the specified direction:
#' - Cell → Nucleus: ligand_obj = cell, receptor_obj = nucleus
#' - Nucleus → Cell: ligand_obj = nucleus, receptor_obj = cell
#' @param ligand_obj A CellChat object representing the source of ligands.
#'                   This can be either a cytoplasm or a nucleus. It must contain
#'                   the relevant ligand-receptor database populated with `@LR` and `@dat.signalling`  for the interaction.
#' @param receptor_obj A CellChat object representing the target of receptors.
#'                     This can be either a cell or a nucleus. It must contain
#'                   the relevant ligand-receptor database populated with `@LR` and `@dat.signalling`  for the interaction.
#' @details
#'
#' @return A list with the communication probabilities and pvalues
#'
cyt_nuc = function(
    ligand_obj = ligand_obj,
    receptor_obj = receptor_obj
) {
  type = 'truncatedMean'
  trim = 0.1 
  LR.use = NULL
  raw.use = TRUE
  population.size = TRUE
  distance.use = TRUE
  interaction.range = 100
  scale.distance = 1
  k.min = 10
  contact.dependent = TRUE
  contact.range = 50
  contact.knn.k = NULL
  contact.dependent.forced = FALSE
  do.symmetric = TRUE
  nboot = 20
  seed.use = 1L
  Kh = 0.5
  n = 1
  FunMean = function(x) mean(x, trim = trim, na.rm = TRUE)
  # change the object to the ligand and receptor data 
  dataL <- as.matrix(ligand_obj@data.signaling)
  dataR <- as.matrix(receptor_obj@data.signaling)
  if (identical(ligand_obj@LR$LRsig, receptor_obj@LR$LRsig)){
    pairLR.use <- ligand_obj@LR$LRsig
  } else {
    print("LR database was different between the receptor and ligand matrix and hence using the ligand matrix to compute the LR pair")
    pairLR.use <- ligand_obj@LR$LRsig
  }
  
  # complex and cofactor inputs (I think I need complex for receptor complex but not sure about cofactor)
  # since database is same it does not matter but better to check 
  if (identical(ligand_obj@DB$complex, receptor_obj@DB$complex)){
    complex_input <- ligand_obj@DB$complex
    cofactor_input <- ligand_obj@DB$cofactor
  } else {
    # fill later
  }
  
  # speeding up the computation 
  if (future::nbrOfWorkers() == 1) {
    my.sapply <- sapply
  } else {
    my.sapply <- future.apply::future_sapply
  }
  
  # sys time
  ptm = Sys.time()
  
  pairLRsig <- pairLR.use
  
  # check the groups how to use them 
  groupL <- ligand_obj@idents
  groupR <- receptor_obj@idents
  
  # genes
  geneL <- as.character(pairLRsig$ligand)
  geneR <- as.character(pairLRsig$receptor)
  
  # number of LR pairs
  nLR <- nrow(pairLRsig)
  
  if (nlevels(groupL) == nlevels(groupR)){
    numCluster <- nlevels(groupL)
  } else {
    # fill later
  }
  
  if (numCluster != length(unique(groupL))) {
    stop("Please check `unique(object@idents)` and ensure that the factor levels are correct!
           You may need to drop unused levels using 'droplevels' function. e.g.,
           `meta$labels = droplevels(meta$labels, exclude = setdiff(levels(meta$labels),unique(meta$labels)))`")
  }
  
  # normalizing
  data.useL <- dataL/max(dataL)
  data.useR <- dataR/max(dataR)
  
  # number of columns use only dataL as the same is assigned to ligand and receptor in dataRavg2 <- dataLavg2
  nCL <- ncol(data.useL)
  nCR <- ncol(data.useR)
  nC <- ncol(data.useL)
  
  # compute the average expression per group - ligand and receptor separately
  data.useL.avg <- aggregate(t(data.useL), list(groupL), FUN = FunMean)
  data.useL.avg <- t(data.useL.avg[,-1])
  colnames(data.useL.avg) <- levels(groupL)
  data.useR.avg <- aggregate(t(data.useR), list(groupR), FUN = FunMean)
  data.useR.avg <- t(data.useR.avg[,-1])
  colnames(data.useR.avg) <- levels(groupR)
  
  # compute the expression of ligand or receptor taking the complex input into account
  dataLavg <- computeExpr_LR(geneL, data.useL.avg, complex_input)
  dataRavg <- computeExpr_LR(geneR, data.useR.avg, complex_input)
  
  # take account into the effect of co-activation and co-inhibition receptors
  dataRavg.co.A.receptor <- computeExpr_coreceptor(cofactor_input, data.useR.avg, pairLRsig, type = "A")
  dataRavg.co.I.receptor <- computeExpr_coreceptor(cofactor_input, data.useR.avg, pairLRsig, type = "I")
  dataRavg <- dataRavg * dataRavg.co.A.receptor/dataRavg.co.I.receptor
  
  # create a df where each row is a normalizing the number of cells by total number of cells
  dataLavg2 <- t(replicate(nrow(dataLavg), as.numeric(table(groupL))/nC))
  dataRavg2 <- dataLavg2
  
  # spatial constraints 
  data.spatialL <- ligand_obj@images$coordinates
  data.spatialR <- receptor_obj@images$coordinates
  ratio <- 1 # this is constant for all the objects 
  tol <- 5
  
  meta.tL = data.frame(group = groupL, samples = ligand_obj@meta$samples, row.names = rownames(ligand_obj@meta))
  meta.tR = data.frame(group = groupR, samples = receptor_obj@meta$samples, row.names = rownames(receptor_obj@meta))
  resL <- computeRegionDistance(coordinates = data.spatialL, meta = meta.tL, interaction.range = interaction.range, ratio = ratio, tol = tol, k.min = k.min, contact.dependent = contact.dependent, contact.range = contact.range, contact.knn.k = contact.knn.k)
  resR <- computeRegionDistance(coordinates = data.spatialR, meta = meta.tR, interaction.range = interaction.range, ratio = ratio, tol = tol, k.min = k.min, contact.dependent = contact.dependent, contact.range = contact.range, contact.knn.k = contact.knn.k)
  d.spatialL <- resL$d.spatial
  d.spatialR <- resR$d.spatial
  adj.contactL <- resL$adj.contact
  adj.contactR <- resR$adj.contact
  d.spatial_list = list(d.spatialL, d.spatialR)
  adj.contactlist = list(adj.contactL, adj.contactR)
  # compute means of the distances (distances are really close)
  d.spatial = Reduce("+", d.spatial_list) / length(d.spatial_list)
  adj.contact = round(Reduce("+", adj.contactlist)/ length(d.spatial_list)) # since the adj.contact is a binary convert anything to rounding
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
  
  Prob <- array(0, dim = c(numCluster,numCluster,nLR))
  Prob_P2_P3_P4 <- array(0, dim = c(numCluster,numCluster,nLR))
  Pval <- array(0, dim = c(numCluster,numCluster,nLR))
  PvalP2_P3_P4 <- array(0, dim = c(numCluster,numCluster,nLR))
  set.seed(seed.use)
  
  # Permutation
  permutationL <- replicate(nboot, sample.int(nCL, size = nCL))
  data.use.avg.bootL <- my.sapply(
    X = 1:nboot,
    FUN = function(nE) {
      groupbootL <- groupL[permutationL[, nE]]
      data.use.avgB <- aggregate(t(data.useL), list(groupbootL), FUN = FunMean)
      data.use.avgB <- t(data.use.avgB[,-1])
      return(data.use.avgB)
    },
    simplify = FALSE
  )
  permutationR <- replicate(nboot, sample.int(nCR, size = nCR))
  data.use.avg.bootR <- my.sapply(
    X = 1:nboot,
    FUN = function(nE) {
      groupbootR <- groupR[permutationR[, nE]]
      data.use.avgB <- aggregate(t(data.useR), list(groupbootR), FUN = FunMean)
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
      
      #  interaction scores
      P2_P3_P4_ablated = P1*P.spatial
      
      # return the matrix of interaction scores
      Prob_P2_P3_P4[ , , i] <- P2_P3_P4_ablated
      
      P2_P3_P4_ablated <- as.vector(P2_P3_P4_ablated)
      
      # here is the calculation of each permutation
      Pboot_ablated <- sapply(
        X = 1:nboot,
        FUN = function(nE) {
          data.use.avgBL <- data.use.avg.bootL[[nE]]
          data.use.avgBR <- data.use.avg.bootR[[nE]]
          dataLavgBL <- computeExpr_LR(geneL[i], data.use.avgBL, complex_input)
          dataRavgBR <- computeExpr_LR(geneR[i], data.use.avgBR, complex_input)
          # take account into the effect of co-activation and co-inhibition receptors
          dataRavgB.co.A.receptor <- computeExpr_coreceptor(cofactor_input, data.use.avgBR, pairLRsig[i, , drop = FALSE], type = "A")
          dataRavgB.co.I.receptor <- computeExpr_coreceptor(cofactor_input, data.use.avgBR, pairLRsig[i, , drop = FALSE], type = "I")
          dataRavgBR <- dataRavgBR * dataRavgB.co.A.receptor/dataRavgB.co.I.receptor
          dataLRB = Matrix::crossprod(dataLavgBL, dataRavgBR)
          P1.boot <- dataLRB^n/(Kh^n + dataLRB^n)
          PbootP2_P3_P4 = as.vector(P1.boot*P.spatial)
          ablatedboot = list("PbootP2_P3_P4"= PbootP2_P3_P4)
          return(ablatedboot)
        }
      )
      # P2 P3 and P4 
      Pboot = Pboot_ablated
      Pboot <- matrix(unlist(Pboot), nrow=length(P2_P3_P4_ablated), ncol = nboot, byrow = FALSE)
      nReject <- rowSums(Pboot - P2_P3_P4_ablated > 0)
      p = nReject/nboot
      PvalP2_P3_P4[, , i] <- matrix(p, nrow = numCluster, ncol = numCluster, byrow = FALSE)
    }
    setTxtProgressBar(pb = pb, value = i)
  }
  close(con = pb)
  PvalP2_P3_P4[Prob_P2_P3_P4 == 0] <- 1
  dimnames(Prob_P2_P3_P4) <- dimnames(Prob) 
  dimnames(PvalP2_P3_P4) <- dimnames(Prob) 
  net <- list("Prob_P2_P3_P4" = Prob_P2_P3_P4,
              "PvalP2_P3_P4" = PvalP2_P3_P4
  )
  return (net)
}

data_dir = 'data/communication_scores'
files = list.files(data_dir) # get the communication_scores calculated object

# cytoplasm to nucleus
for (i in 1:length(files)){
  file = files[i]
  object = readRDS(file.path(data_dir, file))
  net = cyt_nuc(ligand_obj = object$cyt, receptor_obj = object$nuc)
  saveRDS(net, file = file.path('data/processed/cyt_to_nuc', file))
}

# nucleus to cytoplasm
for (i in 1:length(files)){
  file = files[i]
  object = readRDS(file.path(data_dir, file))
  net = cyt_nuc(ligand_obj = object$nuc, receptor_obj = object$cyt)
  saveRDS(net, file = file.path('data/processed/nuc_to_cyt', file))
}

# assign the dimnames
files = list.files(data_dir)
cyt_nuc = 'data/processed/cyt_to_nuc/'
for (i in 1:length(files)){
  file = files[i]
  dat = readRDS(file.path(data_dir, file))
  obj = readRDS(file.path(cyt_nuc, file))
  ligand_obj = dat$cyt
  group <- ligand_obj@idents
  pairLRsig <- ligand_obj@LR$LRsig
  dimnames(obj$Prob_P2_P3_P4) = list(levels(group), levels(group), rownames(pairLRsig))
  dimnames(obj$PvalP2_P3_P4) = list(levels(group), levels(group), rownames(pairLRsig))
  saveRDS(obj, file = file.path('data/processed/cyt_to_nuc', file))
}

nuc_cyt = 'data/processed/nuc_to_cyt/'
for (i in 1:length(files)){
  file = files[i]
  dat = readRDS(file.path(data_dir, file))
  obj = readRDS(file.path(nuc_cyt, file))
  ligand_obj = dat$nuc
  group <- ligand_obj@idents
  pairLRsig <- ligand_obj@LR$LRsig
  dimnames(obj$Prob_P2_P3_P4) = list(levels(group), levels(group), rownames(pairLRsig))
  dimnames(obj$PvalP2_P3_P4) = list(levels(group), levels(group), rownames(pairLRsig))
  saveRDS(obj, file = file.path('data/processed/nuc_to_cyt', file))
}
