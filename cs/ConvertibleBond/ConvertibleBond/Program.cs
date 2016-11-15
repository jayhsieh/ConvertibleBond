using System;
using System.ComponentModel;
using System.IO;
using System.Linq;

namespace LCP
{
    class Program
    {
        static void Main(string[] args)
        {
            ConvertibleBondTest();
            //AmericanPutOptionTest();
        }

        static public void ConvertibleBondTest()
        {
            double sigma = 0.2;
            double rc = 0.02;
            double r = 0.05;
            double rg = 0.05;
            double[] coupon_dates = new double[10];
            linspace(coupon_dates, 0.5, 5, 10);
            double Bc_clean = 110;
            double Bp_clean = 105;
            double Bc = Bc_clean;
            double Bp = Bp_clean;
            double[] CallInterval =  new double[2] {1, 5};
            double[] PutInterval =  new double[2] {2, 3};
            double F  = 101;
            double Coupon = 4;
            double kappa = 1.0;
            double T = 5;
            int N = 201;
            double[] CB = new double[N];
            double[] COCB = new double[N];
            double[] x = new double[N];
            linspace(x, 0, 5*F, N);

            BlkSch[] bs = new BlkSch[2];
            //construct Black Scholes equation for CB
            BoundaryCondition[] CB_bc = new BoundaryCondition[2];
            CB_bc[0] = new BoundaryCondition(); CB_bc[1] = new BoundaryCondition();
            CB_bc[0].bc_type = BoundaryCondition.BC_Type.ZeroS;
            CB_bc[1].bc_type = BoundaryCondition.BC_Type.Dirichlet;
            CB_bc[1].bc_values[0] = kappa * x.Last(); 
            double[] minus_rcB = new double[N];
            bs[0] = new BlkSch(CB, x, 0.5*sigma*sigma, rg, -r, minus_rcB, CB_bc);

            //construct Black Scholes equation for COCB
            BoundaryCondition[] COCB_bc = new BoundaryCondition[2];
            COCB_bc[0] = new BoundaryCondition(); COCB_bc[1] = new BoundaryCondition();
            COCB_bc[0].bc_type = BoundaryCondition.BC_Type.ZeroS;
            COCB_bc[1].bc_type = BoundaryCondition.BC_Type.Dirichlet;
            COCB_bc[1].bc_values[0] = 0;
            bs[1] = new BlkSch(COCB, x, 0.5 * sigma * sigma, rg, -(r + rc), null, COCB_bc);

            //setup constrain
            Action<double[][], int> constrain = new Action<double[][], int>((V, i) => {
                double[] cb = V[0];
                double[] cocb = V[1];
                //upside constrain due to callability
                if (cb[i] > Math.Max(Bc, kappa * x[i])) {
                    cb[i] = Math.Max(Bc, kappa * x[i]);
                    cocb[i] = 0;
                }

                //downside constrain due to puttability  
                if (cb[i] < Bp) {
                    cb[i] = Bp;
                    cocb[i] = Bp;
                }

                //upside constrain due to conversion
                if (cb[i] < kappa * x[i]) {
                    cb[i] = kappa * x[i];
                    cocb[i] = 0;
                }
            });

            //setup initial condition
            for (int i = 0; i < N; ++i) {
                CB[i] = (kappa * x[i] >= F + Coupon) ? kappa * x[i] : F + Coupon;
                COCB[i] = (kappa * x[i] >= F + Coupon) ? 0.0 : F + Coupon;       
            }

            print(CB);
            print(x);
            //construct solver
            BS_PSOR solver = new BS_PSOR(bs, constrain);

            //advance solution backwards from maturity to t = 0;
            double t = T;
            double dt = 1.0 / 365;
            int coupon_index = coupon_dates.Length - 1;
            string CBoutput = "CBsol.txt";
            string COCBoutput = "COCBsol.txt";
            System.IO.File.WriteAllText(CBoutput, string.Empty);
            System.IO.File.WriteAllText(COCBoutput, string.Empty);
            solver.printSolution(CBoutput, CB);
            solver.printSolution(COCBoutput, COCB);
            Console.WriteLine("t = {0}, dt = {1}", t, dt);
            while (t > 0) {
                if (t - dt < 0) dt = t;
                //update source term
                for (int i = 0; i < N; ++i)
                    minus_rcB[i] = -rc * COCB[i];

                //update coefficients

                //update call price
                if (PutInterval != null && t >= PutInterval[0] && t <= PutInterval[1])
                    Bp = Bp_clean + AccI(t, coupon_dates, Coupon);
                else
                    Bp = 0;

                //update put price
                if (CallInterval != null && t >= CallInterval[0] && t <= CallInterval[1])
                    Bc = Bc_clean + AccI(t, coupon_dates, Coupon);
                else
                    Bc = double.MaxValue;

                //advance solution
                solver.advance(dt);

                //compute Coupon between [t-dt, t];
                while (coupon_index >= 0 && coupon_dates[coupon_index] >= t - dt && coupon_dates[coupon_index] <= t) {
                    Console.WriteLine("Paying coupon {0} at {1}", Coupon, coupon_dates[coupon_index]);
                    for (int i = 0; i < CB.Length; ++i) {
                        CB[i] += Coupon;
                        COCB[i] += Coupon;
                    }
                    coupon_index--;
                }
                t = t - dt;
                Console.WriteLine("t = {0}, dt = {1}", t, dt);
                Console.WriteLine("price(100) = {0}", solver.getPrice(bs[0], 100));
                solver.printSolution(CBoutput, CB);
                solver.printSolution(COCBoutput, COCB);
            }
        }

        static private double AccI(double t, double[] coupon_dates, double Coupon) {
            for (int i = 1; i < coupon_dates.Length; ++i)
            {
                if (t >= coupon_dates[i - 1] && t <= coupon_dates[i]) {
                    return Coupon * (t - coupon_dates[i-1])/(coupon_dates[i] - coupon_dates[i-1]);
                }
            }
            return 0;
        }

        static public void AmericanPutOptionTest()
        {
            double sigma = 0.8;
            double E = 100;
            double T = 0.25;
            double dt = 2e-4;
            double r = 0.1;
            int N = 800;
            double[] sol = new double[N];
            double[] x = new double[N];
            linspace(x, 0, 500, N);

            //construct BlackScholes equaiton
            BlkSch[] bs = new BlkSch[1];
            BoundaryCondition[] bc = new BoundaryCondition[2];
            bc[0] = new BoundaryCondition(); bc[1] = new BoundaryCondition();
            bc[0].bc_type = BoundaryCondition.BC_Type.Dirichlet;
            bc[0].bc_values[0] = Math.Max(E - x[0], 0);

            //bc[1].bc_type = BoundaryCondition.BC_Type.Dirichlet;
            bc[1].bc_type = BoundaryCondition.BC_Type.LargeS;
            bc[1].bc_values[0] = Math.Max(E - x.Last(), 0);

            for (int i = 0; i < sol.Length; ++i)
                sol[i] = Math.Max(E - x[i], 0);

            Action<double[][], int> constrain = new Action<double[][], int>((V, i) =>
            {
                V[0][i] = Math.Max(V[0][i], Math.Max(E - x[i], 0));
            });
            bs[0] = new BlkSch(sol, x, sigma*sigma/2, r, -r, null, bc);
            BS_PSOR bs_solver = new BS_PSOR(bs, constrain, BS_PSOR.Scheme.CrankNicolson);

            double t = T;
            while (t > 0) {
                Console.WriteLine("t = {0}", t);
                if (t - dt < 0)
                    dt = t;
                t = t - dt;
                bs_solver.advance(dt);
                bs_solver.printSolution("AmPut.txt", sol);
                Console.WriteLine("price(100) = {0}", bs_solver.getPrice(bs[0], 100));
            }
        }

        static void linspace(double[] x, double smin, double smax, int N) {
            double dx = (smax - smin) / (N-1);
            for (int i = 0; i < N; ++i) {
                x[i] = smin + i * dx;
            }
        }
        static void print(double[] x) {
            Console.WriteLine(string.Join(",", x));
        }
    }

    class BoundaryCondition
    {
        public enum BC_Type { ZeroS, LargeS, Dirichlet};
        public BC_Type bc_type = 0;
        public double[] bc_values = new double [4];
        public BoundaryCondition() {
        }
    }


    /// <summary>
    /// parameters for Black-Scholes equation
    /// </summary>
    class BlkSch {
        public double[] sol;
        public double[] x;
        public double c2;
        public double c1;
        public double c0;
        public double[] source;
        public BoundaryCondition[] bc;

        public BlkSch(
                double[] sol,
                double[] x,
                double s2d2Vd2s,
                double sdVds,
                double V,
                double[] source,
                BoundaryCondition[] bc) {
            this.sol = sol;
            this.x = x;
            this.source = source;
            this.bc = bc;
            c2 = s2d2Vd2s;
            c1 = sdVds;
            c0 = V;
        }

    }
    /// <summary>
    /// solve constrained Black-Scholes equation using Crank-Nicolson Method with
    /// Projected SOR method for linear equation
    /// Input are coefficients for s^2ddV/dds, sdV/ds, V and source term
    /// and boundary conditions
    /// solving direction is backwards from maturity towards time equal to zero
    /// </summary>
    class BS_PSOR {
        //for fully coupled equations
        public BlkSch[] bs;

        //for numerical scheme
        private double theta;   //0 for explicit, 1 for implicit, 0.5 for Crank Nicolson
        private double dt;
        private double min_dx;

        //for linear solver
        Action<double[][], int> constrain;
        enum DIR {UPPER, LOWER};

        public enum Scheme { Explicit, Implicit, CrankNicolson};
        public BS_PSOR(
                BlkSch[] bs,
                Action<double[][], int> constrain,
                Scheme scheme = Scheme.CrankNicolson)
        {
            
            this.constrain = constrain;
            this.bs = bs;
            min_dx = 1000;
            double[] x = bs[0].x;
            for (int i = 0; i < x.Length - 1; ++i) {
                min_dx = Math.Min(min_dx, x[i+1] - x[i]);
            }
            switch (scheme) {
                case Scheme.CrankNicolson:
                    theta = 0.5;
                    break;
                case Scheme.Explicit:
                    theta = 0;
                    break;
                case Scheme.Implicit:
                    theta = 1;
                    break;
                default:
                    theta = 0.5;
                    break;
            }
        }

        public void constructLinearSystem(BlkSch BS, SparseLinearSystem SPS) {
            BoundaryCondition[] bc = BS.bc;
            double[] rhs = SPS.b;
            double c2 = BS.c2;
            double c1 = BS.c1;
            double c0 = BS.c0;
            double[] x = BS.x;
            double[] sol = BS.sol;
            SPS.solution = BS.sol;
            double[] source = BS.source;
            int n = BS.sol.Length;
            applyBoundaryCondition(BS, bc[0], SPS.lower_bc, rhs, x, DIR.LOWER);
            applyBoundaryCondition(BS, bc[1], SPS.upper_bc, rhs, x, DIR.UPPER);
            for (int i = 1; i < n - 1; ++i)
            {
                double alpha = 2 * c2 * x[i] * x[i] / ((x[i] - x[i - 1]) * (x[i + 1] - x[i - 1])) - c1 * x[i] / (x[i + 1] - x[i - 1]);
                double beta = 2 * c2 * x[i] * x[i] / ((x[i + 1] - x[i]) * (x[i + 1] - x[i - 1])) + c1 * x[i] / (x[i + 1] - x[i - 1]);
                SPS.L[i - 1] = -alpha * theta * dt;
                SPS.D[i - 1] = 1 + (alpha + beta - c0) * theta * dt;
                SPS.R[i - 1] = -beta * theta * dt;
                rhs[i] = sol[i - 1] * alpha * (1 - theta) * dt
                       + sol[i] * (1 - (alpha + beta - c0) * (1 - theta) * dt)
                       + sol[i + 1] * beta * (1 - theta) * dt;
                if (source != null)
                    rhs[i] += source[i] * dt;
            }
        }

        public void advance(double dt) {
            this.dt = dt;
            SparseLinearSystem[] sps = new SparseLinearSystem[bs.Length];
            for (int i = 0; i < bs.Length; ++i)
            {
                sps[i] = new SparseLinearSystem(bs[i].sol.Length);
                constructLinearSystem(bs[i], sps[i]);
            }

            PSOR psor_solver = new PSOR(sps, 1.55, constrain);
            psor_solver.tolerance = Math.Min(dt, min_dx) * 0.01;
            psor_solver.maxIter = 200;
            psor_solver.solve();
            //Console.WriteLine("Iteration: {0}, Error Norm: {1}", psor_solver.NumIter, psor_solver.ErrorNorm);
        }

        void applyBoundaryCondition(BlkSch bs, BoundaryCondition bc, double[] coef, double[] rhs, double[] x, DIR dir) {
            switch (bc.bc_type) {
                case BoundaryCondition.BC_Type.ZeroS:
                    if (dir == DIR.UPPER)
                        throw new InvalidEnumArgumentException("Cannot use zero boundary on upper side");
                    coef[0] = 1 - bs.c0 * theta * dt;
                    rhs[0]  = (1 + bs.c0 * (1 - theta) * dt) * bs.sol[0];
                    if (bs.source != null)
                        rhs[0] += bs.source[0] * dt;
                    break;
                case BoundaryCondition.BC_Type.LargeS:
                    if (dir == DIR.LOWER)
                    {
                        throw new InvalidEnumArgumentException("Cannot use large s boundary on lower side");
                    }
                    else
                    {
                        int m = x.Length-1;
                        rhs[m] = 0;
                        coef[m - 2] = -1; coef[m - 1] = 2; coef[m] = -1; 
                    }
                    break;
                case BoundaryCondition.BC_Type.Dirichlet:
                    if (dir == DIR.LOWER)
                    {
                        rhs[0] = bc.bc_values[0];
                        coef[0] = 1;
                    }
                    else
                    {
                        int m = x.Length - 1;
                        rhs[m] = bc.bc_values[0];
                        coef[m] = 1;
                    }
                    break;
            }
        }

        public void printSolution(string fname, double[] sol) {
            using (var stream = new StreamWriter(fname, true))
            {
                foreach (var item in sol)
                {
                    stream.Write("{0:F3} ", item);
                }
                stream.Write(Environment.NewLine);
            }
        }

        public double getPrice(BlkSch bs, double s) {
            double[] sol = bs.sol;
            double[] x = bs.x;
            for (int i = 1; i < sol.Length; ++i) {
                if (x[i-1] <= s && x[i] >= s)
                    return (x[i] - s) / (x[i] - x[i - 1]) * sol[i - 1] 
                         + (s - x[i-1]) / (x[i] - x[i - 1]) * sol[i];
            }
            return 0;
        }
    }

    class SparseLinearSystem {
        public double[] L;
        public double[] R;
        public double[] D;
        public double[] b;
        public double[] lower_bc;
        public double[] upper_bc;
        public double[] solution;
        public SparseLinearSystem(int n) {
            L = new double[n-2];
            D = new double[n-2];
            R = new double[n-2];
            b = new double[n];
            lower_bc = new double[n];
            upper_bc = new double[n];
        }

    }
    /// <summary>
    /// Projected successive over-relaxation method for solving Ax = b with constrains,
    /// A is nxn matrix, b is nx1 vector, x is nx1 vector
    /// This class is only designed for 1D problem,
    /// hence the inputs are tri-diagonal band matrix A, right-hand side b,
    /// and relaxiation factor, constrain on the solution
    /// </summary>
    class PSOR
    {
        SparseLinearSystem[] sls;
        private double w;
        private Action<double[][], int> applyConstrain;
        private double[][] x_new;
        private double[][] x_old;

        public double tolerance = 1e-8;
        public double maxIter = 100;
        public double ErrorNorm { get; private set; }
        public int NumIter { get; private set; }

        public PSOR(SparseLinearSystem[] sparseLinearSystem, double RelaxFactor, Action<double[][], int> constrain) {
            this.sls = sparseLinearSystem;
            w = RelaxFactor;
            applyConstrain = constrain;
            if (x_old == null)
            {
                x_old = new double[sls.Length][];
                for (int i = 0; i < x_old.Length; ++i)
                    x_old[i] = new double[sls[i].solution.Length];
            }
            if (x_new == null)
            {
                x_new = new double[sls.Length][];
                for (int i = 0; i < x_new.Length; ++i)
                    x_new[i] = sls[i].solution;
            }

            if (RelaxFactor < 0 || RelaxFactor > 2)
                throw new ArgumentOutOfRangeException("Relaxation Factor can only between 0 and 2");
            for (int i = 0; i < sls.Length; ++i) {
                foreach (double di in sls[i].D) {
                    if (di == 0)
                        throw new ArgumentException("There are zeros in diagnal");
                }
                if (sls[i].lower_bc[0] == 0 || sls[i].upper_bc.Last() == 0)
                    throw new ArgumentException("There are zeros in diagnal");
            }
        }

        public void doOneIteration() {
            int n = x_new[0].Length;
            for (int sys_id = 0; sys_id < sls.Length; ++sys_id)
                Array.Copy(x_new[sys_id], x_old[sys_id], x_new[sys_id].Length);
            for (int i = 0; i < n; ++i) {
                for (int sys_id = 0; sys_id < sls.Length; ++sys_id)
                {
                    double[] lower_bc = sls[sys_id].lower_bc;
                    double[] upper_bc = sls[sys_id].upper_bc;
                    double[] x = sls[sys_id].solution;
                    double[] b = sls[sys_id].b;
                    double[] D = sls[sys_id].D;
                    double[] L = sls[sys_id].L;
                    double[] R = sls[sys_id].R;
                    if (i == 0)
                    {
                        //first row
                        //Console.WriteLine("Before x[0] = {0}", x[0]);
                        double tmp = 0;
                        for (int j = i + 1; j < n; ++j)
                            tmp += lower_bc[j] * x[j];
                        x[i] = (1 - w) * x[i]
                             + w / lower_bc[i] * (b[i] - tmp);
                        //Console.WriteLine("After x[0] = {0}", x[0]);
                    }
                    else if (i == n - 1)
                    {
                        //last row
                        double tmp = 0;
                        for (int j = 0; j < i; ++j)
                            tmp += upper_bc[j] * x[j];
                        x[i] = (1 - w) * x[i]
                                 + w / upper_bc[i] * (b[i] - tmp);
                    }
                    else
                    {
                        x[i] = (1 - w) * x[i]
                             + w / D[i - 1] * (b[i] - L[i - 1] * x[i - 1] - R[i - 1] * x[i + 1]);
                    }
                }
                if (applyConstrain != null)
                    applyConstrain.Invoke(x_new, i);
            }
            //Console.WriteLine(string.Join(",", D));
        }

        public void solve()
        {
            NumIter = 0;
            ErrorNorm = 1000;
            while (ErrorNorm > tolerance && NumIter++ < maxIter) {
                doOneIteration();
                ErrorNorm = computeError(x_new, x_old);
               // Console.WriteLine("Iteration {0}, error norm = {1}", NumIter, ErrorNorm);
            }
            if (ErrorNorm > tolerance)
            {
                //safeguard
                Console.WriteLine("Warning: iteration limit reached, error norm = {0}", ErrorNorm);
                //throw new WarningException("iteration limit reached");
            }
        }

        private double computeError(double[][] sol_set, double[][] sol_set_old)
        {
            double max_err = 0;
            for (int i = 0; i < sol_set.Length; ++i)
            {
                double[] x = sol_set[i];
                double[] y = sol_set_old[i];
                double err = 0;
                for (int j = 0; j < x.Length; ++j)
                    err += (x[j] - y[j]) * (x[j] - y[j]);
                err = Math.Sqrt(err);
                max_err = Math.Max(max_err, err);
            }
            return max_err;
        }

        private void printArray(double[] array)
        {
            Console.WriteLine(string.Join(",", array));
        }
    }
}
