# Concise Exam Notes: Matrix and Tensor Computations

> Converted from the supplied PDF. Mathematical expressions are preserved with Unicode symbols and readable notation so they display directly in Markdown without raw LaTeX source.

English translation of the Russian page-resolved reference guide. For each exam question, the guide gives the initial intuition, the minimum expected for the exam, and reading references with specific page ranges.

Notation note: mathematical expressions are written as readable Unicode/plain-text notation, not as raw LaTeX source. For example, transposes are shown with ᵀ, inverses with ⁻¹, and approximate equality with ≈.

Important note on pages. Page ranges are given according to the printed pagination of articles and the cited book editions. In article PDFs, the PDF page number may differ by one or two pages because of title or front-matter pages, but the ranges point to the sections where each topic is treated. For preparation from scratch, first read the basic pages, then the specialized articles for the specific question.

## How to Read This Document

- For each question, first read the “Introduction from scratch” block: it sets the meaning and vocabulary.
- Then check the “Exam minimum” block: this lists what you should be able to reproduce orally or in writing.
- In the “Reading with pages” block, the first sources usually give the foundations, while the later ones give specialized theory or algorithms.
- If time is short, read at least the first and second source for each question.

## Table of Questions

### I. Matrix Approximation Methods

- 1. QR with column pivoting and rank-revealing QR
- 2. Randomized SVD
- 3. The Lanczos method
- 4. Recompression of low-rank matrices in skeleton format
- 5. The maximum-volume principle
- 6. The maxvol algorithm
- 7. Cross methods
- 8. Estimating the approximation norm and the error norm in cross methods

### II. Canonical Tensor Decomposition

- 1. Existence of the canonical decomposition and canonical rank
- 2. Unfolding matrices and their connection with canonical rank
- 3. Arithmetic with tensors in canonical format
- 4. Typical ranks and generic rank
- 5. Example of a sequence of tensors converging to a tensor of higher rank
- 6. Kruskal ranks and the d-dimensional generalization of Kruskal’s theorem
- 7. ALS for approximating a tensor by a tensor in canonical format
- 8. ALS for approximating a CP tensor by a lower-rank tensor
- 9. Levenberg-Marquardt method for CP approximation and CP compression
- 10. ALS and Levenberg-Marquardt for approximating Tucker and TT tensors by a CP tensor

### III. Tucker Decomposition

- 1. Existence of Tucker decomposition, minimal decomposition, and Tucker ranks
- 2. HOSVD, st-HOSVD, and approximation error estimates
- 3. Arithmetic in Tucker format and orthogonal Tucker decomposition
- 4. Orthogonalization and quasi-optimal recompression in Tucker format

### IV. Tensor Train

- 1. Existence of the tensor train
- 2. Unfolding matrices for the tensor train
- 3. Minimal TT and the connection between TT ranks and unfolding ranks
- 4. Arithmetic in tensor-train format
- 5. Orthogonality and orthogonalization in tensor-train format
- 6. TT-SVD, quasi-optimal TT recompression, and error estimate
- 7. Idea of the TT-cross method and TT-cross without adaptive rank search
- 8. Adaptive rank search in TT: ideas, problems, and solutions
- 9. Fast elementwise operations on TT tensors through a special version of TT-cross

---

## I. Matrix Approximation Methods

### 1. QR with column pivoting and rank-revealing QR

#### Introduction from scratch

First recall that a QR decomposition replaces the original columns by an orthonormal basis Q and an upper-triangular matrix R. Column pivoting adds a permutation P: the algorithm tries to move the most informative columns to the front. Rank-revealing QR is used when the diagonal of R and the residual block R₂₂ should reveal the numerical rank.

#### Exam minimum

- Be able to write A P = Q R and its block form with R₁₁, R₁₂, and R₂₂.
- Explain the difference between ordinary QR with column pivoting and strong rank-revealing QR.
- Connect a small R₂₂ block and good conditioning of R₁₁ with the quality of the selected subspace.

#### Reading with pages

- [TB97, pp. 50-83] - QR, orthogonal transformations, and least squares from scratch.
- [GVL13, pp. 248-288] - Householder QR, column pivoting, and numerical rank.
- [GE96, pp. 848-856] - formulation of rank-revealing QR and why QR with column pivoting is not always rank-revealing.
- [GE96, pp. 856-869] - strong rank-revealing QR, permutations, and quality estimates.

### 2. Randomized SVD

#### Introduction from scratch

If SVD finds the best k-dimensional singular subspaces, randomized SVD first builds a random sketch of the range of A, usually by multiplying A by a random test matrix Ω. The sketch is then orthonormalized, and an ordinary SVD is performed only on a small matrix. The point is to replace expensive work with A by cheaper work with its randomly discovered range.

#### Exam minimum

- Know the randomized range finder, oversampling parameter p, power iterations, and the approximation A ≈ Q Qᵀ A.
- Understand where the error enters.
- Explain why slowly decaying singular values require power iterations.

#### Reading with pages

- [TB97, pp. 25-45] - SVD, singular values, and best matrix approximation from scratch.
- [HMT11, pp. 217-226] - motivation for randomized low-rank approximation and the error model.
- [HMT11, pp. 227-241] - randomized range finder and basic algorithms.
- [HMT11, pp. 242-260] - power iterations, single-pass variants, and error estimates.

### 3. The Lanczos method

#### Introduction from scratch

The Lanczos method builds a small tridiagonal or bidiagonal representation of a large matrix through Krylov subspaces. In low-rank approximation, it is used to compute a few eigenvalues or singular values without forming a full SVD.

#### Exam minimum

- Distinguish symmetric Lanczos from Golub-Kahan bidiagonalization.
- Understand the three-term recurrence, loss of orthogonality, and the connection with partial SVD.

#### Reading with pages

- [TB97, pp. 235-267] - Krylov subspaces and Arnoldi/Lanczos from scratch.
- [L50, pp. 255-265] - the original Lanczos method and the three-term recurrence.
- [GK65, pp. 205-214] - bidiagonalization for singular values.
- [GVL13, pp. 486-507] - practical SVD computation and Lanczos bidiagonalization.

### 4. Recompression of low-rank matrices in skeleton format

#### Introduction from scratch

A skeleton representation stores a matrix through selected rows and columns, in the form A ≈ C U R. After arithmetic operations, the rank often grows, so recompression is needed: build a more compact representation with nearly the same accuracy.

#### Exam minimum

- Understand the skeleton/CUR formula.
- Explain why the selected submatrix should be nonsingular.
- Describe how SVD or maxvol helps reduce the rank and how the error norm is controlled.

#### Reading with pages

- [TB97, pp. 25-45] - SVD and rank truncation as the basic recompression mechanism.
- [GZT97, pp. 515-519] - pseudo-skeleton approximation and the role of maximum volume.
- [T00, pp. 367-374] - cross/skeleton representation and incomplete approximation.
- [GOSTZ10, pp. 247-256] - practical search for a good submatrix for recompression.

### 5. The maximum-volume principle

#### Introduction from scratch

The volume of a square submatrix is the absolute value of its determinant. The maxvol principle says that if the central submatrix has large volume, then the interpolation coefficients are not too large, and the skeleton/CUR approximation is more stable.

#### Exam minimum

- Explain that an r by r submatrix is selected.
- Explain why the determinant is connected with conditioning.
- Describe how maxvol gives quasi-optimal estimates for cross/skeleton approximation.

#### Reading with pages

- [GZT97, pp. 515-519] - pseudo-skeleton formulation through maximum-volume matrices.
- [GOSTZ10, pp. 247-250] - definition of volume and its connection with a good submatrix.
- [T00, pp. 367-371] - how maxvol enters cross approximation.
- [S14, pp. 217-224] - tensor generalization of maxvol interpolation.

### 6. The maxvol algorithm

#### Introduction from scratch

The maxvol algorithm is a greedy local search for a submatrix of large volume. Usually one starts with some r by r submatrix and then replaces rows or columns until the coefficients in A C⁻¹ become sufficiently small.

#### Exam minimum

- Know the local stopping condition.
- Understand the meaning of the coefficient matrix.
- Explain why global maxvol is expensive to find.
- Describe how local maxvol is used in cross approximation and TT-cross.

#### Reading with pages

- [GOSTZ10, pp. 247-256] - algorithm for finding a good submatrix and its applications.
- [GZT97, pp. 515-519] - theoretical background for pseudo-skeleton approximation and maxvol.
- [S14, pp. 224-236] - maxvol in nested tensor indices.
- [OT10, pp. 78-84] - use of maxvol in TT-cross sweeps.

### 7. Cross methods

#### Introduction from scratch

Cross approximation builds an approximation from a small number of entries, rows, and columns, without scanning the whole matrix. This is especially important when an element A(i,j) can be computed quickly, but the full matrix is too large.

#### Exam minimum

- Be able to write the cross approximation using selected columns, the inverse of the selected intersection submatrix, and selected rows: A is approximated by the product of those three blocks.
- Explain the choice of row and column index sets I and J.
- Explain the relation with adaptive cross approximation and the difference from SVD, which requires access to the whole matrix.

#### Reading with pages

- [T00, pp. 367-380] - incomplete cross approximation and the mosaic-skeleton method.
- [B00, pp. 565-575] - adaptive cross approximation for boundary-element matrices.
- [GZT97, pp. 515-519] - pseudo-skeleton error for a good choice of submatrix.
- [GOSTZ10, pp. 247-256] - practical selection of rows and columns through maxvol.

### 8. Estimating the approximation norm and the error norm in cross methods

#### Introduction from scratch

In cross methods, one cannot simply inspect the full residual because the full object is often unavailable. Therefore one uses a posteriori estimates: test entries, stabilization of selected fibers or sections, estimates through maxvol, and the norm of interpolation coefficients.

#### Exam minimum

- Distinguish the norm of the approximation itself from the norm of the error.
- Understand why the error depends on the quality of the selected submatrix.
- Be able to describe a practical stopping criterion.

#### Reading with pages

- [GZT97, pp. 515-519] - estimates for pseudo-skeleton approximation.
- [T00, pp. 371-380] - criteria and analysis of incomplete cross approximation.
- [B00, pp. 575-589] - practical estimates and adaptivity of adaptive cross approximation.
- [S14, pp. 224-244] - quasi-optimality of maxvol interpolation in the tensor case.

## II. Canonical Tensor Decomposition

### 1. Existence of the canonical decomposition and canonical rank

#### Introduction from scratch

The canonical, or CP/PARAFAC, decomposition expresses a tensor as a sum of rank-one tensors. The canonical rank is the minimum number of such terms. Unlike the matrix rank, tensor rank is difficult: best approximations of a prescribed rank may fail to exist.

#### Exam minimum

- Know the CP form, rank-one outer products, CP rank, the difference from matrix rank, and the non-closedness of sets of bounded CP rank.

#### Reading with pages

- [KB09, pp. 455-464] - tensor notation, outer products, and matricization from scratch.
- [KB09, pp. 464-469] - CP decomposition and tensor rank.
- [CC70, pp. 283-292] - classic introduction to CANDECOMP.
- [H70, pp. 1-15, 47-58] - classic introduction to PARAFAC and its factor interpretation.
- [DSL08, pp. 1084-1092] - why the best low-rank tensor approximation may not exist.

### 2. Unfolding matrices and their connection with canonical rank

#### Introduction from scratch

An unfolding turns a tensor into a matrix by grouping indices. For a CP decomposition, each mode unfolding is a factor matrix multiplied by the Khatri-Rao product of the remaining factors. Therefore the matrix ranks of unfoldings give lower bounds for the CP rank.

#### Exam minimum

- Be able to write the matricized CP form using the factor matrix in mode n, the diagonal weights, and the Khatri-Rao product of the other factors.
- Explain matricization, the Khatri-Rao product, and inequalities between matrix ranks and CP ranks.

#### Reading with pages

- [KB09, pp. 459-464] - matricization, n-mode product, Kronecker product, and Khatri-Rao product.
- [KB09, pp. 464-469] - CP in matricized form.
- [DL00, pp. 1256-1264] - n-mode vectors and unfoldings as the basis of multilinear algebra.
- [TB97, pp. 25-45] - SVD and matrix rank as the basic support.

### 3. Arithmetic with tensors in canonical format

#### Introduction from scratch

The CP format stores a tensor as a sum of components. Addition simply concatenates components, while elementwise and tensor products usually increase the rank. The practical problem in CP arithmetic is growth in the number of components and the need for later compression.

#### Exam minimum

- Know that storage is proportional to d · n · R.
- Know how CP tensors are added.
- Know how scalar products are computed through Gram matrices.
- Explain why recompression is needed after operations.

#### Reading with pages

- [KB09, pp. 459-464] - tensor operations and Khatri-Rao/Kronecker notation.
- [KB09, pp. 464-473] - CP representation, ALS, and practical computations.
- [O11, pp. 2308-2311] - contrast with TT arithmetic and the problem of rank growth.
- [ADK11, pp. 67-76] - computation of the CP objective and gradients in an optimization formulation.

### 4. Typical ranks and generic rank

#### Introduction from scratch

For real tensors of the same size, several typical ranks may exist: open sets of tensors can have different ranks. The generic rank describes the rank of almost all tensors over the complex field, or the most typical situation in a chosen model.

#### Exam minimum

- Distinguish maximal rank, typical rank, and generic rank.
- Explain why matrices do not have this pathology, while tensor rank depends on the field and on structure.

#### Reading with pages

- [KB09, pp. 470-473] - typical ranks and special features of tensor rank.
- [DSL08, pp. 1084-1094] - geometry of rank and pathologies of low-rank sets.
- [K77, pp. 95-104] - ranks of three-dimensional arrays and early estimates.
- [H70, pp. 58-70] - interpretation of PARAFAC rank in factor models.

### 5. Example of a sequence of tensors converging to a tensor of higher rank

#### Introduction from scratch

The central counterexample is that a sequence of tensors of rank r may converge to a tensor of rank greater than r. This shows that the set of tensors with rank at most r need not be closed and that the best CP approximation of a prescribed rank may not exist.

#### Exam minimum

- Be able to explain the idea of border rank: a limiting object can have higher ordinary CP rank but lower border rank.
- Connect this with diverging components in numerical CP algorithms.

#### Reading with pages

- [DSL08, pp. 1084-1101] - explicit examples of ill-posed best low-rank tensor approximation.
- [KB09, pp. 470-473] - concise discussion of border rank and CP-rank problems.
- [TB05, pp. 163-170] - practical connection with degeneracy and missing values in PARAFAC.
- [K77, pp. 95-102] - basic definitions of rank for three-way arrays.

### 6. Kruskal ranks and the d-dimensional generalization of Kruskal’s theorem

#### Introduction from scratch

The Kruskal rank of a factor matrix is the largest k such that any k of its columns are linearly independent. Kruskal’s theorem gives a sufficient condition for uniqueness of a CP decomposition through the sum of the k-ranks of the factor matrices.

#### Exam minimum

- Know the k-rank.
- Know the three-mode condition kA + kB + kC ≥ 2R + 2.
- Understand the admissible ambiguities: permutation of components and scaling of factors.

#### Reading with pages

- [K77, pp. 95-104] - definitions of rank, k-rank, and the uniqueness problem.
- [K77, pp. 105-121] - proof idea and uniqueness theorem.
- [SB00, pp. 229-239] - N-way/generalized uniqueness conditions.
- [KB09, pp. 469-473] - modern presentation of CP uniqueness and the Kruskal condition.

### 7. ALS for approximating a tensor by a tensor in canonical format

#### Introduction from scratch

ALS fixes all factor matrices except one and solves a linear least-squares problem for the remaining factor. Repeating this over the modes gives a simple and popular method for CP approximation, but it may converge slowly and fall into poor local minima.

#### Exam minimum

- Be able to derive the normal equations through the Khatri-Rao product and MTTKRP.
- Understand component normalization, stopping criteria, and the degeneracy problem.

#### Reading with pages

- [KB09, pp. 464-473] - CP model, matricized CP, and the ALS algorithm.
- [ADK11, pp. 67-76] - optimization formulation of CP and computation of the gradient/MTTKRP.
- [ADKM11, pp. 41-48] - weighted CP and sparse/incomplete data.
- [TB97, pp. 84-107] - least squares and normal equations from scratch.

### 8. ALS for approximating a CP tensor by a lower-rank tensor

#### Introduction from scratch

Here the input tensor is already given in CP form with high rank, and the goal is to find a CP representation of lower rank. The main idea is to compute ALS steps and scalar products through the factors of the source CP tensor and the target CP tensor, without expanding the full array.

#### Exam minimum

- Understand CP compression, rank growth after operations, and computation of the norm of the difference of two CP tensors through Gram matrices.
- Explain why the problem remains nonlinear and nonconvex.

#### Reading with pages

- [KB09, pp. 464-473] - CP representation, ALS, and Gram/Khatri-Rao notation.
- [ADK11, pp. 67-78] - CP as nonlinear least squares and computations without the full tensor.
- [HMT11, pp. 217-226] - matrix analogy for low-rank compression.
- [O11, pp. 2308-2311] - comparison with rounding in stable formats.

### 9. Levenberg-Marquardt method for CP approximation and CP compression

#### Introduction from scratch

CP approximation is a nonlinear least-squares problem in the factor matrices. Levenberg-Marquardt adds damping to the Gauss-Newton step, helping the method move between a fast Newton-like step and stable gradient-descent behavior.

#### Exam minimum

- Know the residual, the Jacobian, the damped normal equations (JᵀJ + μI) Δ = -Jᵀr, why damping is needed, and which block structures arise from CP factors.

#### Reading with pages

- [TB97, pp. 84-107] - least squares from scratch and the origin of normal equations.
- [KB09, pp. 464-473] - CP objective and ALS as basic optimization.
- [ADK11, pp. 67-86] - CP as a scalable optimization problem, gradients, and Hessian approximations.
- [TB05, pp. 163-180] - Levenberg-Marquardt for PARAFAC with missing values.

### 10. ALS and Levenberg-Marquardt for approximating Tucker and TT tensors by a CP tensor

#### Introduction from scratch

The source tensor may be given implicitly, in Tucker or TT form. Then CP fitting must compute objective values, gradients, and ALS/LM steps through contractions of compact cores without forming the full array.

#### Exam minimum

- Explain how the Tucker core or TT cores enter MTTKRP and the gradient.
- Identify which parameters determine the cost.
- Explain why conversion between formats can be useful for subsequent arithmetic.

#### Reading with pages

- [KB09, pp. 464-473] - CP objective, ALS, and Khatri-Rao notation.
- [KB09, pp. 473-480] - Tucker model and matricized forms.
- [O11, pp. 2297-2311] - TT cores, unfoldings, rounding, and operations.
- [ADK11, pp. 67-86] - nonlinear optimization viewpoint useful for LM-style CP fitting.

## III. Tucker Decomposition

### 1. Existence of Tucker decomposition, minimal decomposition, and Tucker ranks

#### Introduction from scratch

The Tucker decomposition represents a tensor as a small core multiplied along each mode by a factor matrix. Unlike CP, an exact Tucker decomposition always exists, and the Tucker rank is the vector of ranks of the mode unfoldings.

#### Exam minimum

- Know the Tucker form: a core tensor G multiplied by factor matrices along all modes.
- Know multilinear rank, the minimal Tucker core, and the connection between Tucker ranks and unfolding ranks.

#### Reading with pages

- [KB09, pp. 459-464] - n-mode product and matricization from scratch.
- [KB09, pp. 473-478] - Tucker decomposition, core tensor, and multilinear rank.
- [T66, pp. 279-288] - original three-mode factor-analysis model.
- [DL00, pp. 1256-1264] - n-mode vectors, ranks, and HOSVD preliminaries.

### 2. HOSVD, st-HOSVD, and approximation error estimates

#### Introduction from scratch

HOSVD chooses the leading left singular vectors of the mode unfoldings and projects the tensor onto those subspaces. Sequentially truncated HOSVD performs truncations one after another, so the later SVDs are smaller. The error is controlled by the sum of discarded tails over the modes.

#### Exam minimum

- Be able to describe the HOSVD algorithm.
- Explain how it differs from the optimal Tucker approximation.
- Know the quasi-optimal estimate and practical rank selection.

#### Reading with pages

- [DL00, pp. 1256-1268] - HOSVD theorem, computation by mode unfoldings, and properties.
- [KB09, pp. 476-478] - compact HOSVD algorithm and quasi-best approximation comment.
- [VVM12, pp. A1027-A1038] - error expression and motivation for sequential truncation.
- [VVM12, pp. A1038-A1052] - ST-HOSVD algorithm, rank truncation, and experiments.

### 3. Arithmetic in Tucker format and orthogonal Tucker decomposition

#### Introduction from scratch

In Tucker format, arithmetic reduces to operations with small cores and factor matrices. If the factors are orthonormal, the tensor norm equals the norm of the core, and projections and errors become simpler.

#### Exam minimum

- Know addition through block expansion of the core.
- Know mode products for applying operators.
- Know scalar products through small contractions.
- Understand the role of factor orthonormality.

#### Reading with pages

- [KB09, pp. 473-480] - Tucker notation, core, factor matrices, and matricized forms.
- [DL00, pp. 1258-1268] - all-orthogonality, norm, and SVD-like properties of HOSVD.
- [VVM12, pp. A1027-A1034] - orthogonal Tucker approximation and error formula.
- [TB97, pp. 1-45] - orthogonality, norms, and SVD from scratch.

### 4. Orthogonalization and quasi-optimal recompression in Tucker format

#### Introduction from scratch

Orthogonalization transfers non-orthogonality of the factor matrices into the core, usually by QR factorizations along the modes. The small core can then be recompressed by HOSVD or ST-HOSVD, and the factors updated, giving cheap recompression without forming the full tensor.

#### Exam minimum

- Describe the QR transfer into the core.
- Explain why orthogonal form is convenient for norms.
- Explain how the error tolerance is distributed across modes.
- Explain what quasi-optimal means.

#### Reading with pages

- [DL00, pp. 1260-1268] - orthogonal HOSVD structure and norm properties.
- [KB09, pp. 476-480] - truncated Tucker/HOSVD and computation.
- [VVM12, pp. A1027-A1052] - T-HOSVD, ST-HOSVD, and error-controlled truncation.
- [GVL13, pp. 248-288] - QR/SVD machinery used in orthogonalization.

## IV. Tensor Train

### 1. Existence of the tensor train

#### Introduction from scratch

Tensor Train, or TT, represents a d-dimensional tensor as a chain of three-dimensional cores. A TT representation exists for every finite tensor: it is constructed by successive SVDs of unfoldings that separate the first k indices from the rest.

#### Exam minimum

- Know the TT formula: the entry with indices i1, ..., id is obtained by multiplying the corresponding core slices G₁(i₁) G₂(i₂) … Gd(id).
- Know the boundary ranks r0 = rd = 1.
- Know that storage is linear in d when TT ranks are moderate.

#### Reading with pages

- [KB09, pp. 455-464] - basic tensor notation and unfoldings.
- [O11, pp. 2295-2298] - definition of TT format and motivation.
- [O11, pp. 2298-2301] - theorem of existence via unfoldings and sequential SVD.
- [TB97, pp. 25-45] - SVD as the matrix foundation of TT construction.

### 2. Unfolding matrices for the tensor train

#### Introduction from scratch

For TT, the important matrices are not only mode unfoldings. One uses matrices whose rows are indexed by a left block of indices (i₁,…,ik) and whose columns are indexed by the remaining right block (i(k+1), ..., id). The ranks of these matrices are exactly the minimal TT ranks.

#### Exam minimum

- Be able to describe the unfolding Ak as the reshaping of the tensor into a matrix of size (n1 · ... · nk) by (n(k+1) · ... · nd).
- Explain why there are d − 1 such unfoldings and how they are used in TT-SVD.

#### Reading with pages

- [O11, pp. 2297-2299] - unfolding matrices Ak and rank definitions.
- [O11, pp. 2299-2301] - constructive proof by sequential decompositions.
- [KB09, pp. 459-464] - general matricization/unfolding background.
- [TB97, pp. 25-45] - matrix rank and SVD from scratch.

### 3. Minimal TT and the connection between TT ranks and unfolding ranks

#### Introduction from scratch

The minimal TT ranks equal the ranks of the unfolding matrices that split the indices into left and right blocks. The representation is not unique because of gauge freedom between neighboring cores, but the minimal ranks are well defined.

#### Exam minimum

- Know the theorem rk = rank(Ak).
- Explain why sets of bounded TT ranks are closed.
- Explain how TT differs from CP in the existence of a best approximation.

#### Reading with pages

- [O11, pp. 2297-2301] - theorem connecting TT ranks with unfolding ranks.
- [O11, pp. 2301-2306] - approximation and rounding, stability of TT ranks.
- [DSL08, pp. 1084-1092] - contrast with CP rank and ill-posed low-rank approximation.
- [KB09, pp. 470-473] - tensor-rank pathologies for comparison.

### 4. Arithmetic in tensor-train format

#### Introduction from scratch

TT arithmetic uses local operations with cores. Addition adds ranks, the Hadamard product multiplies ranks, and scalar products are computed by sequential contractions. After operations, TT-rounding is almost always needed.

#### Exam minimum

- Know formulas or constructions for the sum, scalar product, and Hadamard product.
- Understand rank growth and costs of the form proportional to d · n · r³, or similar estimates.

#### Reading with pages

- [O11, pp. 2308-2311] - addition, scalar product, Hadamard product, and contraction in TT.
- [O11, pp. 2302-2306] - rounding used after rank growth.
- [HRS12, pp. A683-A695] - TT optimization context and local tensor operations.
- [KB09, pp. 455-464] - tensor operation notation.

### 5. Orthogonality and orthogonalization in tensor-train format

#### Introduction from scratch

TT cores can be made left- or right-orthogonal by QR or SVD after reshaping local cores into matrices. A mixed-canonical form chooses an orthogonality center and makes norm computation, local optimization, and rank truncation stable.

#### Exam minimum

- Be able to write the reshaping of a core into a matrix.
- Explain left and right orthogonality, gauge transformations, and transfer of the R factor into the neighboring core.

#### Reading with pages

- [O11, pp. 2302-2306] - orthogonalization and the TT-rounding algorithm.
- [HRS12, pp. A683-A695] - ALS in TT and the role of orthogonal/canonical forms.
- [DS14, pp. A2248-A2258] - rank-adaptive TT/AMEn setting and basis enrichment.
- [GVL13, pp. 248-288] - QR machinery behind TT orthogonalization.

### 6. TT-SVD, quasi-optimal TT recompression, and error estimate

#### Introduction from scratch

TT-SVD performs truncated SVDs of unfoldings successively, moving the remaining part to the right. For a tensor already given in TT form, recompression is performed by orthogonalization and local SVDs without forming the full array. The total error is the square root of the sum of local tail errors squared.

#### Exam minimum

- Know the TT-SVD algorithm.
- Know how to distribute the tolerance δ.
- Know the estimate: the Frobenius norm of A − B is at most the square root of the sum of the squared local errors.
- Know the quasi-optimal factor √(d − 1).

#### Reading with pages

- [O11, pp. 2298-2302] - exact and approximate TT-SVD with the error theorem.
- [O11, pp. 2302-2306] - TT-rounding for tensors already in TT format.
- [TB97, pp. 25-45] - Eckart-Young/SVD background.
- [VVM12, pp. A1027-A1038] - useful analogy with sequential Tucker truncation.

### 7. Idea of the TT-cross method and TT-cross without adaptive rank search

#### Introduction from scratch

TT-cross builds a TT approximation from individual entries of the array, like matrix cross approximation, but with nested multi-index sections. If the ranks are known in advance, the algorithm performs sweeps and updates selected fixed-size index sets.

#### Exam minimum

- Know black-box access to entries, nested index sets, and local maxvol choices.
- Explain the difference between fixed-rank TT-cross and TT-SVD, which requires the full tensor.

#### Reading with pages

- [OT10, pp. 70-76] - TT-cross motivation and interpolation formula.
- [OT10, pp. 76-84] - fixed-rank sweeps and maxvol index updates.
- [GOSTZ10, pp. 247-256] - maxvol submatrix search used inside cross methods.
- [S14, pp. 217-230] - nested interpolation sets and quasi-optimality background.

### 8. Adaptive rank search in TT: ideas, problems, and solutions

#### Introduction from scratch

When TT ranks are unknown, the method must increase them as needed. Problems include local errors that may not reflect the global error, poor sections that spoil maxvol, and ranks that may grow too fast. Solutions include DMRG/MALS, enrichment, AMEn, and external error checks.

#### Exam minimum

- Explain adaptive rank selection, two-site updates, residual/enrichment vectors, stopping criteria, and the tradeoff between stability and cost.

#### Reading with pages

- [OT10, pp. 80-88] - rank-adaptation issues in the TT-cross setting.
- [HRS12, pp. A683-A713] - ALS/MALS optimization in TT format.
- [DS14, pp. A2248-A2271] - AMEn, enrichment, and rank adaptivity for high-dimensional systems.
- [S14, pp. 230-244] - greedy/nested cross interpolation and error behavior.

### 9. Fast elementwise operations on TT tensors through a special version of TT-cross

#### Introduction from scratch

For a nonlinear operation f(A) or g(A,B), direct computation of all entries is impossible, while formal TT arithmetic can sharply increase the ranks. A special TT-cross treats the result as a black box: an entry of the result is computed quickly from the original TT tensors, and cross approximation builds a compact TT representation of the result.

#### Exam minimum

- Understand which elementwise operations can be computed on request.
- Explain why rank adaptation is important.
- Explain how maxvol indices choose informative entries.
- Know when this is preferable to explicit arithmetic.

#### Reading with pages

- [O11, pp. 2308-2311] - elementwise/Hadamard product and rank growth in TT.
- [OT10, pp. 70-88] - TT-cross for black-box multidimensional arrays.
- [S14, pp. 217-244] - quasi-optimal maximum-volume cross interpolation for tensors.
- [DS14, pp. A2258-A2267] - enrichment/adaptivity ideas useful for nonlinear black-box outputs.

## Bibliography

Full source records used in the page-specific references above.
- **[ADK11]** Acar, E., Dunlavy, D. M., Kolda, T. G. A scalable optimization approach for fitting canonical tensor decompositions. Journal of Chemometrics 25(2), 67-86, 2011. DOI: 10.1002/cem.1335.
- **[ADKM11]** Acar, E., Dunlavy, D. M., Kolda, T. G., Morup, M. Scalable tensor factorizations for incomplete data. Chemometrics and Intelligent Laboratory Systems 106(1), 41-56, 2011. DOI: 10.1016/j.chemolab.2010.08.004.
- **[B00]** Bebendorf, M. Approximation of boundary element matrices. Numerische Mathematik 86, 565-589, 2000.
- **[CC70]** Carroll, J. D., Chang, J.-J. Analysis of individual differences in multidimensional scaling via an N-way generalization of Eckart-Young decomposition. Psychometrika 35, 283-319, 1970.
- **[DL00]** De Lathauwer, L., De Moor, B., Vandewalle, J. A multilinear singular value decomposition. SIAM J. Matrix Anal. Appl. 21(4), 1253-1278, 2000. DOI: 10.1137/S0895479896305696.
- **[DS14]** Dolgov, S. V., Savostyanov, D. V. Alternating minimal energy methods for linear systems in higher dimensions. SIAM J. Sci. Comput. 36(5), A2248-A2271, 2014. DOI: 10.1137/140953289.
- **[DSL08]** de Silva, V., Lim, L.-H. Tensor rank and the ill-posedness of the best low-rank approximation problem. SIAM J. Matrix Anal. Appl. 30(3), 1084-1127, 2008. DOI: 10.1137/06066518X.
- **[GE96]** Gu, M., Eisenstat, S. C. Efficient algorithms for computing a strong rank-revealing QR factorization. SIAM J. Sci. Comput. 17(4), 848-869, 1996. DOI: 10.1137/0917055.
- **[GK65]** Golub, G., Kahan, W. Calculating the singular values and pseudo-inverse of a matrix. SIAM J. Numer. Anal. Series B 2(2), 205-224, 1965.
- **[GOSTZ10]** Goreinov, S. A., Oseledets, I. V., Savostyanov, D. V., Tyrtyshnikov, E. E., Zamarashkin, N. L. How to find a good submatrix. In Matrix Methods: Theory, Algorithms and Applications, World Scientific, 247-256, 2010.
- **[GVL13]** Golub, G. H., Van Loan, C. F. Matrix Computations. 4th ed. Johns Hopkins University Press, 2013.
- **[GZT97]** Goreinov, S. A., Zamarashkin, N. L., Tyrtyshnikov, E. E. Pseudo-skeleton approximations by matrices of maximal volume. Mathematical Notes 62(4), 515-519, 1997. DOI: 10.1007/BF02358985.
- **[H70]** Harshman, R. A. Foundations of the PARAFAC procedure: models and conditions for an explanatory multimodal factor analysis. UCLA Working Papers in Phonetics 16, 1-84, 1970.
- **[HMT11]** Halko, N., Martinsson, P.-G., Tropp, J. A. Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review 53(2), 217-288, 2011. DOI: 10.1137/090771806.
- **[HRS12]** Holtz, S., Rohwedder, T., Schneider, R. The alternating linear scheme for tensor optimization in the tensor train format. SIAM J. Sci. Comput. 34(2), A683-A713, 2012. DOI: 10.1137/100818893.
- **[K77]** Kruskal, J. B. Three-way arrays: rank and uniqueness of trilinear decompositions. Linear Algebra and Its Applications 18(2), 95-138, 1977. DOI: 10.1016/0024-3795(77)90069-6.
- **[KB09]** Kolda, T. G., Bader, B. W. Tensor decompositions and applications. SIAM Review 51(3), 455-500, 2009. DOI: 10.1137/07070111X.
- **[L50]** Lanczos, C. An iteration method for the solution of the eigenvalue problem of linear differential and integral operators. J. Research NBS 45(4), 255-282, 1950.
- **[O11]** Oseledets, I. V. Tensor-train decomposition. SIAM J. Sci. Comput. 33(5), 2295-2317, 2011. DOI: 10.1137/090752286.
- **[OT10]** Oseledets, I. V., Tyrtyshnikov, E. E. TT-cross approximation for multidimensional arrays. Linear Algebra and Its Applications 432(1), 70-88, 2010. DOI: 10.1016/j.laa.2009.07.024.
- **[S14]** Savostyanov, D. V. Quasioptimality of maximum-volume cross interpolation of tensors. Linear Algebra and Its Applications 458, 217-244, 2014. DOI: 10.1016/j.laa.2014.06.006.
- **[SB00]** Sidiropoulos, N. D., Bro, R. On the uniqueness of multilinear decomposition of N-way arrays. Journal of Chemometrics 14(3), 229-239, 2000.
- **[T00]** Tyrtyshnikov, E. E. Incomplete cross approximation in the mosaic-skeleton method. Computing 64(4), 367-380, 2000. DOI: 10.1007/s006070070031.
- **[T66]** Tucker, L. R. Some mathematical notes on three-mode factor analysis. Psychometrika 31(3), 279-311, 1966. DOI: 10.1007/BF02289464.
- **[TB05]** Tomasi, G., Bro, R. PARAFAC and missing values. Chemometrics and Intelligent Laboratory Systems 75(2), 163-180, 2005. DOI: 10.1016/j.chemolab.2004.07.003.
- **[TB97]** Trefethen, L. N., Bau, D. Numerical Linear Algebra. SIAM, 1997.
- **[VVM12]** Vannieuwenhoven, N., Vandebril, R., Meerbergen, K. A new truncation strategy for the higher-order singular value decomposition. SIAM J. Sci. Comput. 34(2), A1027-A1052, 2012. DOI: 10.1137/110836067.
