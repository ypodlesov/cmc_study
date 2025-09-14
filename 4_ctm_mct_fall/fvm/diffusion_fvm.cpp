#include <stdio.h>

#include "inmost.h"

using namespace INMOST;
using namespace std;

const double dx = 1.0;
const double dy = 1.0;
const double dxy = 0.0;
const double a = 3.14;

double C(double x, double y)  // analytical solution
{
  // return x;
  return sin(a * x) * sin(a * y);
}

double source(double x, double y) {
  // return 0;
  return -a * a *
         (2. * dxy * cos(a * x) * cos(a * y) -
          (dx + dy) * sin(a * x) * sin(a * y));
}

enum BoundCondType { BC_DIR = 1, BC_NEUM = 2 };

// Class including everything needed
class Problem {
 private:
  /// Mesh
  Mesh &m;
  // =========== Tags =============
  /// Solution tag: 1 real value per cell
  Tag tagConc;
  /// Diffusion tensor tag: 3 real values (Dx, Dy, Dxy) per cell
  Tag tagD;
  /// Boundary condition type tag: 1 integer value per face, sparse on faces
  Tag tagBCtype;
  /// Boundary condition value tag: 1 real value per face, sparse on faces
  Tag tagBCval;
  /// Right-hand side tag: 1 real value per cell
  Tag tagSource;
  /// Analytical solution tag: 1 real value per cell
  Tag tagConcAn;
  /// Global index tag: 1 integer value per cell
  Tag tagGlobInd;

  // =========== Tag names ===========
  const string tagNameConc = "Concentration";
  const string tagNameD = "Diffusion_tensor";
  const string tagNameBCtype = "BC_type";
  const string tagNameBCval = "BC_value";
  const string tagNameSource = "Source";
  const string tagNameConcAn = "Concentration_analytical";
  const string tagNameGlobInd = "Global_Index";

 public:
  Problem(Mesh &m_);
  ~Problem();
  void initProblem();
  void assembleGlobalSystem(Sparse::Matrix &A, Sparse::Vector &rhs);
  void run();
};

Problem::Problem(Mesh &m_) : m(m_) {}

Problem::~Problem() {}

void Problem::initProblem() {
  // Init tags
  tagConc = m.CreateTag(tagNameConc, DATA_REAL, CELL, NONE, 1);
  tagD = m.CreateTag(tagNameD, DATA_REAL, CELL, NONE, 3);
  tagBCtype = m.CreateTag(tagNameBCtype, DATA_INTEGER, FACE, FACE, 1);
  tagBCval = m.CreateTag(tagNameBCval, DATA_REAL, FACE, FACE, 1);
  tagSource = m.CreateTag(tagNameSource, DATA_REAL, CELL, CELL, 1);
  tagConcAn = m.CreateTag(tagNameConcAn, DATA_REAL, CELL, CELL, 1);
  tagGlobInd = m.CreateTag(tagNameGlobInd, DATA_INTEGER, CELL, NONE, 1);

  // Cell loop
  // 1. Set diffusion tensor values
  // 2. Write analytical solution and source tags
  // 3. Assign global indices
  int glob_ind = 0;
  for (Mesh::iteratorCell icell = m.BeginCell(); icell != m.EndCell();
       icell++) {
    Cell c = icell->getAsCell();
    c.RealArray(tagD)[0] = dx;   // Dx
    c.RealArray(tagD)[1] = dy;   // Dy
    c.RealArray(tagD)[2] = dxy;  // Dxy
    double xc[2];
    c.Barycenter(xc);
    c.Real(tagConcAn) = C(xc[0], xc[1]);
    c.Real(tagSource) = source(xc[0], xc[1]);
    c.Integer(tagGlobInd) = glob_ind;
    glob_ind++;
  }

  // Face loop:
  // 1. Set BC
  for (Mesh::iteratorFace iface = m.BeginFace(); iface != m.EndFace();
       iface++) {
    Face f = iface->getAsFace();
    if (!f.Boundary()) continue;
    f.Integer(tagBCtype) = BC_DIR;
    double xf[2];
    f.Barycenter(xf);
    f.Real(tagBCval) = C(xf[0], xf[1]);
  }
}

double calc_tf(rMatrix const &D, rMatrix const &nf, double *dA) {
  rMatrix DdA(2, 1);
  double dAnorm = 0, dBnorm = 0;
  DdA(0, 0) = D(0, 0) * dA[0] + D(0, 1) * dA[1];
  DdA(1, 0) = D(1, 0) * dA[0] + D(1, 1) * dA[1];
  double temp = (DdA(0, 0) * nf(0, 0) + DdA(1, 0) * nf(1, 0)) /
                (dA[0] * dA[0] + dA[1] * dA[1]);
  return temp;
}

void Problem::assembleGlobalSystem(Sparse::Matrix &A, Sparse::Vector &rhs) {
  // Face loop
  // Calculate transmissibilities using
  // two-point flux approximation (TPFA)
  for (Mesh::iteratorFace iface = m.BeginFace(); iface != m.EndFace();
       iface++) {
    Face f = iface->getAsFace();
    double xf[2];
    rMatrix nf(2, 1);
    f.UnitNormal(nf.data());
    f.Barycenter(xf);
    if (f.Boundary()) {
      int BCtype = f.Integer(tagBCtype);
      if (BCtype == BC_NEUM) {
      } else if (BCtype == BC_DIR) {
        Cell cA;
        cA = f.BackCell();
        double xA[2];
        cA.Barycenter(xA);
        double dA[2];
        dA[0] = xf[0] - xA[0];
        dA[1] = xf[1] - xA[1];

        rMatrix D(2, 2);
        D(0, 0) = cA.RealArray(tagD)[0];
        D(0, 1) = cA.RealArray(tagD)[2];
        D(1, 0) = cA.RealArray(tagD)[2];
        D(1, 1) = cA.RealArray(tagD)[1];

        int indA = cA.Integer(tagGlobInd);

        rhs[indA] += f.Real(tagBCval) * calc_tf(D, nf, dA);
        A[indA][indA] += calc_tf(D, nf, dA);

        // implement by yourself
      }
    } else {
      // Internal face
      Cell cA, cB;
      cA = f.BackCell();
      cB = f.FrontCell();
      if (!cB.isValid()) {
        cout << "Invalid FrontCell!" << endl;
        exit(1);
      }
      double xA[2], xB[2];
      cA.Barycenter(xA), cB.Barycenter(xB);
      // std::cout << xA[0] << ' ' << xA[1] << std::endl;
      double dA[2], dB[2];
      dA[0] = xf[0] - xA[0];
      dA[1] = xf[1] - xA[1];
      dB[0] = xf[0] - xB[0];
      dB[1] = xf[1] - xB[1];

      rMatrix DA(2, 2);
      DA(0, 0) = cA.RealArray(tagD)[0];
      DA(0, 1) = cA.RealArray(tagD)[2];
      DA(1, 0) = cA.RealArray(tagD)[2];
      DA(1, 1) = cA.RealArray(tagD)[1];

      rMatrix DB(2, 2);
      DB(0, 0) = cA.RealArray(tagD)[0];
      DB(0, 1) = cB.RealArray(tagD)[2];
      DB(1, 0) = cB.RealArray(tagD)[2];
      DB(1, 1) = cB.RealArray(tagD)[1];

      double tf = (calc_tf(DA, nf, dA) * calc_tf(DB, nf, dB)) /
                  (calc_tf(DA, nf, dA) - calc_tf(DB, nf, dB));

      int indA = cA.Integer(tagGlobInd);
      int indB = cB.Integer(tagGlobInd);

      A[indA][indA] += -(tf * f.Area());
      A[indB][indB] += -(tf * f.Area());
      A[indA][indB] += (tf * f.Area());
      A[indB][indA] += (tf * f.Area());
    }
  }
  for (auto icell = m.BeginCell(); icell != m.EndCell(); icell++) {
    Cell c = icell->getAsCell();
    int i = c.Integer(tagGlobInd);
    double center[2];
    c.Barycenter(center);
    rhs[i] += source(center[0], center[1]) * c.Volume();
  }
}

void Problem::run() {
  // Matrix size
  unsigned N = static_cast<unsigned>(m.NumberOfCells());
  // Global matrix called 'stiffness matrix'
  Sparse::Matrix A;
  // Solution vector
  Sparse::Vector sol;
  // Right-hand side vector
  Sparse::Vector rhs;

  A.SetInterval(0, N);
  sol.SetInterval(0, N);
  rhs.SetInterval(0, N);

  assembleGlobalSystem(A, rhs);

  string solver_name = "inner_mptiluc";
  Solver S(solver_name);
  S.SetParameter("drop_tolerance", "0");
  S.SetParameter("absolute_tolerance", "1e-14");
  S.SetParameter("relative_tolerance", "1e-10");

  S.SetMatrix(A);
  bool solved = S.Solve(rhs, sol);
  printf("Number of iterations: %d\n", S.Iterations());
  printf("Residual:             %e\n", S.Residual());
  if (!solved) {
    printf("Linear solver failed: %s\n", S.GetReason().c_str());
    exit(1);
  }

  double normC = 0.0, normL2 = 0.0;
  for (Mesh::iteratorCell icell = m.BeginCell(); icell != m.EndCell();
       icell++) {
    Cell c = icell->getAsCell();
    unsigned ind = static_cast<unsigned>(c.Integer(tagGlobInd));
    c.Real(tagConc) = sol[ind];
    double diff = fabs(c.Real(tagConc) - c.Real(tagConcAn));
    normL2 += diff * c.Volume();
    normC = max(normC, diff);
  }
  printf("Error C-norm:  %e\n", normC);
  printf("Error L2-norm: %e\n", normL2);

  m.Save("res.pvtk");
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s mesh_file\n", argv[0]);
    return -1;
  }

  Mesh m;
  m.Load(argv[1]);
  Problem P(m);
  P.initProblem();
  P.run();
  printf("Success\n");
  return 0;
}
