using System;
using System.Collections.Generic;
using System.Linq;
using Axioma.Pricers.Callable;
using Axioma.Pricers.Convertible;
using System.IO;
using System.ComponentModel;
using CenterSpace.NMath.Matrix;
using CenterSpace.NMath.Core;

namespace Axioma.Phoenix.Platform.MathsLayer.SecurityPricingHelpers.PDEFramework
{

    /// <summary>
    /// Abstract class for convertible bond pricing
    /// </summary>
    public abstract class ConvertibleBondPDEPricer {

        #region public_members
        /// <summary>
        /// user is able to set the time interval for saving results to disk
        /// </summary>
        public double print_time_interval = double.MaxValue;
        /// <summary>
        /// user is able to set the output directory
        /// default is the current directory
        /// rerun the simulation will overwrite the old results
        /// </summary>
        public string OutDir = ".";
        /// <summary>
        /// user can change the grid size of space, default is 200
        /// </summary>
        public int GridSize = 200;
        #endregion

        #region protected_members
        /// <summary>
        /// 
        /// </summary>
        protected double Bc = double.MaxValue, Bp = 0, call_barr = 0;

        /// <summary>
        /// 
        /// </summary>
        protected BlkSch[] BlkScholes;

        /// <summary>
        /// 
        /// </summary>
        protected double[] space;

        /// <summary>
        /// time control variables
        /// </summary>
        protected double t, dt, T;

        /// <summary>
        /// 
        /// </summary>
        protected double[][] solution;

        /// <summary>
        /// 
        /// </summary>
        protected double sigma, kappa, F;

        /// <summary>
        /// 
        /// </summary>
        protected Func<double, double> riskFree, creditSpread, growthRate;
        #endregion

        #region private_members
        private List<double[]> solution_set = new List<double[]>();
        private List<double> time = new List<double>();
        private double min_dx = double.MaxValue;
        ConvertibleBondParameters convertibleBondParams;

        //for dirty price
        bool is_dirty_price = false;
        double? dirty_price;
        double? clean_price;

        //for all schedules
        List<CouponParameters> coupon = null;
        OptionalityPeriodParameters[] callSch = null;
        OptionalityPeriodParameters[] putSch = null;
        BarrierParameters[] callBarrSch = null;

        //time control variables
        bool is_print_time = false;
        int print_time_count = 0;
        int coupon_index = -1;
        int put_index = -1;
        int call_index = -1;
        int call_barr_index = -1;

        // solver for helping computing Vega
        ConvertibleBondPDEPricer solver2;
        #endregion

        #region pure virtual function
        /// <summary>
        /// allocate memory for domain vectors
        /// grids and solution
        /// </summary>
        protected abstract void setDomain();

        /// <summary>
        /// create PDE and associated I.C. and B.C.
        /// </summary>
        protected abstract void createPDE();

        /// <summary>
        /// set initial condition for PDE;
        /// </summary>
        protected abstract void setInitialCondition();

        /// <summary>
        /// constraints for the solution of PDE
        /// </summary>
        protected abstract void constraints(double[][] sol, int index);

        /// <summary>
        /// virtual constructor
        /// </summary>
        /// <returns> a cloned object</returns>
        protected abstract ConvertibleBondPDEPricer clone();
        /// <summary>
        /// compute source term for PDE
        /// </summary>
        protected abstract void computeSource();
        #endregion

        /// <summary>
        /// Constructor for convertible bond pricer using pde model
        /// </summary>
        /// <param name="convertibleBondParams"></param>
        /// <param name="conversionRatio"></param>
        /// <param name="volatility"></param>
        /// <param name="riskFree"></param>
        /// <param name="growthRate"></param>
        /// <param name="creditSpread"></param>
        /// <param name="isDirtyPrice"></param>
        public ConvertibleBondPDEPricer(ConvertibleBondParameters convertibleBondParams,
                                            double conversionRatio,
                                            double volatility,
                                            Func<double, double> riskFree,
                                            Func<double, double> growthRate,
                                            Func<double, double> creditSpread,
                                            bool isDirtyPrice = false) {

            this.convertibleBondParams = convertibleBondParams;
            extractParameters(convertibleBondParams);
            this.sigma = volatility;
            this.kappa = conversionRatio;
            this.riskFree = riskFree;
            this.creditSpread = creditSpread;
            this.growthRate = growthRate;
            is_dirty_price = isDirtyPrice;
        }

        /// <summary>
        /// Shallow copy constructor, solution space is recreated
        /// </summary>
        /// <param name="solver"></param>
        public ConvertibleBondPDEPricer(ConvertibleBondPDEPricer solver)
        {
            convertibleBondParams = solver.convertibleBondParams;
            extractParameters(convertibleBondParams);
            sigma = solver.sigma;
            kappa = solver.kappa;
            riskFree = solver.riskFree;
            creditSpread = solver.creditSpread;
            growthRate = solver.growthRate;
            GridSize = solver.GridSize;
            print_time_interval = solver.print_time_interval;
            is_dirty_price = solver.is_dirty_price;
        }

        private void extractParameters(ConvertibleBondParameters CBparams)
        {
            //sort call schedule
            if (CBparams.CallBarrierSchedule != null)
                callSch = CBparams.CallSchedule.OrderBy(a => a.Start).ToArray();
            //sort put schedule
            if (CBparams.PutSchedule != null)
                putSch = CBparams.PutSchedule.OrderBy(a => a.Start).ToArray();
            //sort coupon schedule
            if (CBparams.Coupons != null)
                coupon = CBparams.Coupons.OrderBy(a => a.PaymentTime).ToList();
            //sort callbarrier
            if (CBparams.CallBarrierSchedule != null)
                callBarrSch = CBparams.CallBarrierSchedule.OrderBy(a => a.End).ToArray();

            T = CBparams.Maturity;

            F = CBparams.FaceValue;
        }

        /// <summary>
        /// Main solver for create and solve pde
        /// </summary>
        public void CreateAndSolvePDE()
        {
            setDomain();

            createPDE();

            setInitialCondition();

            advancePDE();
        }

        private void advancePDE()
        {
            min_dx = computeMinDx(space);

            //set up list index for call, put, coupon
            if (putSch != null)
                put_index = putSch.Length - 1;
            if (callSch != null)
                call_index = callSch.Length - 1;
            if (coupon != null)
                coupon_index = coupon.Count - 1;
            if (callBarrSch != null)
                call_barr_index = callBarrSch.Length - 1;

            //construct solver
            BS_PSOR solver = new BS_PSOR(BlkScholes, constraints, BS_PSOR.Scheme.Implicit);

            t = T;
            dt = min_dx * 1.0e-2;

            string MyDir = this.OutDir + "/Results";
            cleanDirectory(MyDir);

            //advance solution backwards from maturity to t = 0;
            while (t > 0)
            {
                //setup dt
                dt = min_dx * 1e-2;
                timeControlFilter(t, ref dt); //update dt for next step
                t = t - dt;

                computeSource();

                //update put price
                //for discrete put schedule where put Start == End, 
                //t will be exactly at the put End
                while (put_index >= 0 && t < putSch[put_index].Start)
                    put_index--;
                if (put_index >= 0 && t <= putSch[put_index].End)
                {
                    Bp = putSch[put_index].Strike * F + AccI(t, coupon);
                }
                else
                    Bp = 0;

                //update call price
                while (call_index >= 0 && t < callSch[call_index].Start)
                    call_index--;
                if (call_index >= 0 && t <= callSch[call_index].End)
                {
                    Bc = callSch[call_index].Strike * F + AccI(t, coupon);
                }
                else
                    Bc = double.MaxValue;

                //update call barrier
                while (call_barr_index >= 0 && t <= callBarrSch[call_barr_index].Start)
                    call_barr_index--;
                if (call_barr_index >= 0 && t < callBarrSch[call_barr_index].End)
                {
                    call_barr = callBarrSch[call_barr_index].Strike * F;
                }
                else
                    call_barr = 0;

                //advance solution
                solver.advance(t, dt);

                updateSolution();

                saveSolution();

                if (is_print_time)
                {
                    System.Diagnostics.Debug.WriteLine("t = {0}, next_dt = {1}", t, dt);
                    System.Diagnostics.Debug.WriteLine("Iteration: {0}, Error Norm: {1}", solver.NumIter, solver.ErrorNorm);
                    printResults(MyDir, t);
                }
            }
        }

        /// <summary>
        /// update solution: add coupon payments to the solution, 
        /// this function can be overrided
        /// </summary>
        protected virtual void updateSolution()
        {
            //compute Coupon between [t-dt, t], t-dt will be exactly at coupon paytime;
            //notice that t has already been updated by t-dt
            while (coupon_index >= 0 && Math.Abs(coupon[coupon_index].PaymentTime - (t)) < 1e-10)
            {
                System.Diagnostics.Debug.WriteLine("{0} Coupon payments at {1}, t = {2}, dt = {3}",
                    coupon_index, coupon[coupon_index].PaymentTime, t, dt);
                for (int i = 0; i < space.Length; ++i)
                {
                    for (int j = 0; j < solution.Length; ++j)
                        solution[j][i] += coupon[coupon_index].PaymentAmount * F;
                }
                coupon_index--;
            }
        }

        /// <summary>
        /// functions to get price
        /// </summary>
        public double DirtyPrice
        {
            get
            {
                if (!is_dirty_price || dirty_price == null)
                {
                    throw new InvalidOperationException("Dirty price has not been set");
                }
                else
                {
                    return (double)dirty_price;
                }
            }
            set { dirty_price = value; }

        }

        /// <summary>
        /// 
        /// </summary>
        public double CleanPrice
        {
            get
            {
                if (is_dirty_price || clean_price == null)
                {
                    throw new InvalidOperationException("Clean price has not been set");
                }
                else
                {
                    return (double)clean_price;
                }
            }
            set { clean_price = value; }
        }

        /// <summary>
        /// compute greeks using central difference for interior nodes 
        /// and forward and backward difference for boundary points, second order of accuracy in total
        /// </summary>
        /// <param name="S"></param>
        /// <param name="t"></param>
        public double computeGamma(double S, double t)
        {
            double ds = min_dx;
            double S_m = getPrice(S - ds, t);
            double S_p = getPrice(S + ds, t);
            return (S_p + S_m - 2 * S) / (ds * ds);
        }

        /// <summary>
        /// compute greeks using central difference for interior nodes 
        /// and forward and backward difference for boundary points, second order of accuracy in total
        /// </summary>
        /// <param name="S"></param>
        /// <param name="t"></param>
        /// <returns></returns>
        public double computeDelta(double S, double t)
        {
            double ds = min_dx;
            double S_m = getPrice(S - ds, t);
            double S_p = getPrice(S + ds, t);
            return (S_p - S_m) / (2.0 * ds);
        }

        /// <summary>
        /// compute Vega by shifting vega 1 percent
        /// </summary>
        /// <param name="S"></param>
        /// <param name="t"></param>
        /// <returns></returns>
        public double computeVega(double S, double t)
        {
            //make sure every time you change parameters,
            //you will reconstruct the solver
            //so that solver2 will be set to be null
            if (solver2 == null)
            {
                solver2 = this.clone(); //virtual constructor
                solver2.sigma = sigma * 1.01;
                solver2.CreateAndSolvePDE();
            }
            double V_p = solver2.getPrice(S, t);
            double V = this.getPrice(S, t);
            return (V_p - V) / (0.01 * sigma);
        }

        //print results
        private void printResults(string OutDir, double t)
        {
#if DEBUG
            //if directory not exist, create it
            try
            {
                Directory.CreateDirectory(OutDir);
            }
            catch (Exception)
            {
                return;
            }
            string TimeOutput = OutDir + "/time.txt";

            for (int i = 0; i < solution.Length; ++i)
                printSolution(OutDir + "/sol-" + i.ToString() + ".txt", solution[i]);
            printTimeStamp(TimeOutput, t);
#endif
        }

        double computeMinDx(double[] x)
        {
            double res = double.MaxValue;
            for (int i = 1; i < x.Length; ++i)
                res = Math.Min(x[i] - x[i - 1], res);
            return res;
        }
        double computeMaxDt(double dt, List<double[]> sol_list)
        {
            if (sol_list.Count < 2)
                return 1.0 / 365;
            const double D = 1;
            double dnorm = 0.1;
            double[] x = sol_list[sol_list.Count - 1];
            double[] x_old = sol_list[sol_list.Count - 2];
            double min_ratio = double.MaxValue;
            for (int i = 0; i < x.Length; ++i)
            {
                double tmp = dnorm / Math.Abs(x[i] - x_old[i])
                           * Math.Max(D, Math.Max(Math.Abs(x[i]), Math.Abs(x_old[i])));
                min_ratio = Math.Min(min_ratio, tmp);
            }
            return dt * min_ratio;
        }
        //control time step dt
        private void timeControlFilter(double t, ref double dt)
        {
            double dt1 = t - (T - print_time_count * print_time_interval); //for print interval
            double dt2 = t; //for termination
            double dt3 = computeCouponDt(t, dt);//for coupon payments
            double dt4 = computePutDt(t, dt); //put could be discrete
            double new_dt = Math.Min(Math.Min(Math.Min(dt1, dt2), dt3), dt4);

            is_print_time = false;
            if (new_dt > dt)
                return;

            if (Math.Abs(dt1 - new_dt) < 1e-15) //check if reaches print time
            {
                is_print_time = true;
                print_time_count++;
            }
            dt = new_dt;
        }

        private double computePutDt(double t, double dt)
        {
            if (convertibleBondParams.PutSchedule == null)
                return dt;
            var putSched = convertibleBondParams.PutSchedule.ToArray();
            double tmp_dt = dt;
            if (put_index < 0) // no schedule
                tmp_dt = dt;
            else if (t <= putSched[put_index].End) //consider t in the inerval
                tmp_dt = t - putSched[put_index].Start;
            else                                  //consider t before the interval
                tmp_dt = t - putSched[put_index].End;
            return (tmp_dt < 1e-15) ? dt : tmp_dt;
        }

        private double computeCouponDt(double t, double dt)
        {   
            List<CouponParameters> coupon = convertibleBondParams.Coupons;
            if (coupon_index >= 0)
                return t - coupon[coupon_index].PaymentTime;
            else
                return dt;
        }

        /// <summary>
        /// append solution at time t to solution space. 
        /// currently save one solution and one 
        /// </summary>
        protected virtual void saveSolution() {
            //save one time solution to the solution space
            solution_set.Add((double[])solution[0].Clone());
            time.Add(t);
        }

        private double getPrice(double S, double[] sol)
        {
            for (int i = 1; i < space.Length; ++i)
            {
                if (space[i - 1] <= S && space[i] >= S)
                {
                    //interpolate in space
                    return (space[i] - S) / (space[i] - space[i - 1]) * sol[i - 1]
                         + (S - space[i - 1]) / (space[i] - space[i - 1]) * sol[i];
                }
            }
            //default
            return 0;
        }

        private void LinearInterpolate(IList<double> grid, double value, int index, double[] coeff, int[] indicies)
        {
            int j = index;
            int j1, j2;
            double c1, c2;

            int N = grid.Count;
            if (j < 0 && ~j == 0) //leftmost, extrapolate
            {
                j1 = 0;
                j2 = 1;
            }
            else if (j < 0 && ~j >= N - 1) //rightmost, extrapolate
            {
                j1 = N - 1;
                j2 = N - 2;
            }
            else if (j < 0) // interior point, not found, interpolate
            {
                j1 = ~j - 1;
                j2 = ~j;
            }
            else // interior point, found the exact one!
            {
                j1 = j;
                j2 = j;
            }

            if (j1 != j2)
            {
                c1 = (value - grid[j2]) / (grid[j1] - grid[j2]);
                c2 = (grid[j1] - value) / (grid[j1] - grid[j2]);
            }
            else
            {
                c1 = 1.0;
                c2 = 0.0;
            }

            indicies[0] = j1; indicies[1] = j2;
            coeff[0] = c1; coeff[1] = c2;
        }
        /// <summary>
        /// get price of convertible bond at time t with underlying price S
        /// using bilinear interpolation, considering query at boundaries, inside or outside domain
        /// </summary>
        /// <param name="S">uderlying</param>
        /// <param name="t">time</param>
        /// <returns></returns>
        public double getPrice(double S, double t = 0)
        {
            int N = time.Count;
            //search in time grid, time is in descending order
            int j = time.BinarySearch(t, new DescendingOrder());
            double[] t_coef = new double[2];
            int[] t_indx = new int[2];
            LinearInterpolate(time, t, j, t_coef, t_indx);

            //search in space grid
            int i = Array.BinarySearch(space, S);
            double[] s_coef = new double[2];
            int[] s_indx = new int[2];
            LinearInterpolate(space, S, i, s_coef, s_indx);

            double p1 = s_coef[0] * solution_set[t_indx[0]][s_indx[0]] + s_coef[1] * solution_set[t_indx[0]][s_indx[1]];
            double p2 = s_coef[0] * solution_set[t_indx[1]][s_indx[0]] + s_coef[1] * solution_set[t_indx[1]][s_indx[1]];

            //interpolate in time
            return t_coef[0] * p1 + t_coef[1] * p2;
        }

        private void cleanDirectory(string OutDir)
        {
#if DEBUG
            if (!Directory.Exists(OutDir))
                return;
            print_time_count = 0;
            System.IO.DirectoryInfo di = new DirectoryInfo(OutDir);

            foreach (FileInfo file in di.GetFiles())
            {
                file.Delete();
            }
            foreach (DirectoryInfo dir in di.GetDirectories())
            {
                dir.Delete(true);
            }
#endif
        }

        private void printTimeStamp(string outName, double t)
        {
            using (var stream = new StreamWriter(outName, true))
            {
                stream.Write("{0} ", t);
            }
        }

        private void printSolution(string outName, double[] solution) {
            using (var stream = new StreamWriter(outName, true))
            {
                foreach (var item in solution)
                {
                    stream.Write("{0:F3} ", item);
                }
                stream.Write(Environment.NewLine);
            }
        }

        private double AccI(double t, List<CouponParameters> coupons)
        {
            if (is_dirty_price || coupons == null)
                return 0.0;

            double F = convertibleBondParams.FaceValue;
            for (int i = 1; i < coupons.Count; ++i)
            {
                if (t >= coupons[i - 1].PaymentTime && t <= coupons[i].PaymentTime)
                {
                    return coupons[i].PaymentAmount * F
                            * (t - coupons[i - 1].PaymentTime)
                            / (coupons[i].PaymentTime - coupons[i - 1].PaymentTime);
                }
            }
            return 0;
        }

        /// <summary>
        /// create N-vector start from smin to smax
        /// </summary>
        /// <param name="x"></param>
        /// <param name="smin"></param>
        /// <param name="smax"></param>
        /// <param name="N"></param>
        protected void linspace(double[] x, double smin, double smax, int N)
        {
            double dx = (smax - smin) / (N - 1);
            for (int i = 0; i < N; ++i)
            {
                x[i] = smin + i * dx;
            }
        }
        static void print(double[] x)
        {
            System.Diagnostics.Debug.WriteLine(string.Join(",", x));
        }
    }

    /// <summary>
    /// Price convertible bond using TF model, proposed in 
    /// "KosTas Tsiveriotis And Chris Fernandes, Valuing Convertible Bonds with credit risk, 
    /// The Journal of Fixed Income, 1998, 95--102."
    /// </summary>
    public class TFPricer :  ConvertibleBondPDEPricer
    {
        private double[] CB;
        private double[] COCB;
        private double[] minus_rcB;
        /// <summary>
        /// Constructor
        /// </summary>
        public TFPricer(ConvertibleBondParameters convertibleBondParams,
                                            double conversionRatio,
                                            double volatility,
                                            Func<double, double> riskFree,
                                            Func<double, double> growthRate,
                                            Func<double, double> creditSpread,
                                            bool isDirtyPrice = false) 
            : base(convertibleBondParams, conversionRatio, volatility, riskFree, growthRate, creditSpread, isDirtyPrice)
        {
        }

        /// <summary>
        /// copy constructor
        /// </summary>
        /// <param name="pricer"></param>
        public TFPricer(TFPricer pricer) : base(pricer)
        {
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        override protected ConvertibleBondPDEPricer clone()
        {
            return new TFPricer(this);
        }
        /// <summary>
        /// set domain
        /// </summary>
        override protected void setDomain()
        {
            //setup space grid
            CB = new double[GridSize];
            COCB = new double[GridSize];
            space = new double[GridSize];
            minus_rcB = new double[GridSize];
            solution = new double[2][];
            solution[0] = CB;
            solution[1] = COCB;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="V"></param>
        /// <param name="i"></param>
        protected override void constraints(double[][] V, int i)
        {
            double[] cb = V[0];
            double[] cocb = V[1];
            double[] x = space;
            //upside constrain due to callability
            if (space[i] < call_barr) //no call if equity price is below the call barrier
                Bc = double.MaxValue;
            if (cb[i] > Math.Max(Bc, kappa * x[i]))
            {
                cb[i] = Math.Max(Bc, kappa * x[i]);
                cocb[i] = 0;
            }

            //downside constrain due to putability  
            if (cb[i] < Bp)
            {
                cb[i] = Bp;
                cocb[i] = Bp;
            }

            //upside constrain due to conversion
            if (cb[i] < kappa * x[i])
            {
                cb[i] = kappa * x[i];
                cocb[i] = 0;
            }

            //keep positivity
            cb[i] = Math.Max(cb[i], 0.0);
            cocb[i] = Math.Max(cocb[i], 0.0);
        }
        /// <summary>
        /// create PDE and associated B.C.
        /// </summary>
        override protected void createPDE()
        {
            BlkScholes = new BlkSch[2];
            double[] x = space;
            int N = space.Length;
            //construct Black Scholes equation for CB
            BoundaryCondition[] CB_bc = new BoundaryCondition[2];
            CB_bc[0] = new BoundaryCondition(); CB_bc[1] = new BoundaryCondition();
            CB_bc[0].bc_type = BoundaryCondition.BC_Type.ZeroS;
            CB_bc[1].bc_type = BoundaryCondition.BC_Type.LargeS;
            CB_bc[1].bc_values[0] = kappa * x.Last();
            Func<double, double, double> c2 = (t, s) => 0.5 * sigma * sigma;
            Func<double, double, double> c1 = (t, s) => growthRate(t);
            Func<double, double, double> c00 = (t, s) => -riskFree(t);
            BlkScholes[0] = new BlkSch(CB, x, c2, c1, c00, minus_rcB, CB_bc);

            //construct Black Scholes equation for COCB
            BoundaryCondition[] COCB_bc = new BoundaryCondition[2];
            COCB_bc[0] = new BoundaryCondition(); COCB_bc[1] = new BoundaryCondition();
            COCB_bc[0].bc_type = BoundaryCondition.BC_Type.ZeroS;
            COCB_bc[1].bc_type = BoundaryCondition.BC_Type.LargeS;
            COCB_bc[1].bc_values[0] = 0;
            Func<double, double, double> c01 = (t, s) => -(riskFree(t) + creditSpread(t));
            BlkScholes[1] = new BlkSch(COCB, x, c2, c1, c01, null, COCB_bc);
        }

        /// <summary>
        /// setup initial condition
        /// </summary>
        protected override void setInitialCondition()
        {
            //setup initial condition
            linspace(space, 0, 5 * F, space.Length);
            for (int i = 0; i < space.Length; ++i)
            {
                double[] x = space;
                CB[i] = (kappa * x[i] >= F) ? kappa * x[i] : F;
                COCB[i] = (kappa * x[i] >= F) ? 0.0 : F;
            }
        }

        /// <summary>
        /// compute source term for PDE
        /// </summary>
        protected override void computeSource()
        {
            //update source term
            double rc = creditSpread(t);
            for (int i = 0; i < space.Length; ++i)
                minus_rcB[i] = -rc * COCB[i];
        }
    }

    /// <summary>
    /// Price convertible bond using AFV model, proposed in
    /// "E. Ayache, P.A. Forsyth and K.R. Vetzal, The Valuation of Convertible Bonds with Credit Risk,
    /// The Journal of Derivatives, 2003,  9-29"
    /// </summary>
    public class AFVPricer : ConvertibleBondPDEPricer
    {
        double[] CB; //solution
        double[] source; //source term
        double eta;  //default rate: 1 for total default, 0 for partial default
        double R;    //recovery factor
        double X;    //recovery value: Face value or predefault value
        double alpha; //parameter in hazard rate model
        double S0;     //spot price
        double p0;   //hazard rate
        Func<double, double, double> p; //hazard rate func

        /// <summary>
        /// Constructor
        /// </summary>
        public AFVPricer(ConvertibleBondParameters convertibleBondParams,
                                            double spotPrice,
                                            double conversionRatio,
                                            double volatility,
                                            Func<double, double> riskFree,
                                            Func<double, double> growthRate,
                                            Func<double, double> creditSpread,
                                            Func<double, double, double> hazardRateFunc = null,
                                            double hazardRate = 0,
                                            double defaultRate = 0,
                                            double recoveryFactor = 0,
                                            bool isDirtyPrice = false) 
            : base(convertibleBondParams, conversionRatio, volatility, riskFree, growthRate, creditSpread, isDirtyPrice)
        {
            S0 = spotPrice;
            R = recoveryFactor;
            X = F;
            p0 = hazardRate;
            eta = defaultRate;
            alpha = -1.2;
            double eps = 1e-5;
            if (hazardRateFunc != null)
            {
                p = hazardRateFunc;
            }
            else
            {
                if (S0 == 0)
                    throw new DivideByZeroException("spot price cannot be zero");
                p = (t, s) => p0 * Math.Pow( (s + eps)/S0, alpha);
            }
        }

        /// <summary>
        /// allocate memory for grids and solution space
        /// </summary>
        protected override void setDomain()
        {
            //setup space grid
            CB = new double[GridSize];
            space = new double[GridSize];
            source = new double[GridSize];
            solution = new double[1][];
            solution[0] = CB;
        }

        /// <summary>
        /// compute source term or sink term
        /// </summary>
        protected override void computeSource()
        {
            //update source term
            for (int i = 0; i < space.Length; ++i)
                source[i] = p(t, space[i]) * Math.Max(kappa * (1 - eta) *space[i], R * X);
        }

        /// <summary>
        /// setup constraints for projected SOR
        /// </summary>
        /// <param name="V"></param>
        /// <param name="i"></param>
        protected override void constraints(double[][] V, int i)
        {
            double[] cb = V[0];
            double[] x = space;
            //upside constrain due to callability
            if (space[i] < call_barr) //no call if equity price is below the call barrier
                Bc = double.MaxValue;
            if (cb[i] > Math.Max(Bc, kappa * x[i]))
            {
                cb[i] = Math.Max(Bc, kappa * x[i]);
            }

            //downside constrain due to putability  
            if (cb[i] < Bp)
            {
                cb[i] = Bp;
            }

            //upside constrain due to conversion
            if (cb[i] < kappa * x[i])
            {
                cb[i] = kappa * x[i];
            }

            //keep positivity
            cb[i] = Math.Max(cb[i], 0.0);
        }

        /// <summary>
        /// create PDE
        /// </summary>
        protected override void createPDE()
        {
            BlkScholes = new BlkSch[1];
            double[] x = space;
            int N = space.Length;
            //construct Black Scholes equation for CB
            BoundaryCondition[] CB_bc = new BoundaryCondition[2];
            CB_bc[0] = new BoundaryCondition(); CB_bc[1] = new BoundaryCondition();
            CB_bc[0].bc_type = BoundaryCondition.BC_Type.ZeroS;
            CB_bc[1].bc_type = BoundaryCondition.BC_Type.LargeS;
            CB_bc[1].bc_values[0] = kappa * x.Last();
            Func<double, double, double> c2 = (t, s) => 0.5 * sigma * sigma;
            Func<double, double, double> c1 = (t, s) => growthRate(t) + p(t,s)*eta;
            Func<double, double, double> c00 = (t, s) => -(riskFree(t) + p(t,s));
            BlkScholes[0] = new BlkSch(CB, x, c2, c1, c00, source, CB_bc);
        }

        /// <summary>
        /// setup initial condition
        /// </summary>
        protected override void setInitialCondition()
        {
            //setup initial condition
            linspace(space, 0, 5 * F, space.Length);
            for (int i = 0; i < space.Length; ++i)
            {
                double[] x = space;
                CB[i] = (kappa * x[i] >= F) ? kappa * x[i] : F;
            }
            System.Diagnostics.Debug.WriteLine(string.Join(",", CB));
        }

        private AFVPricer(AFVPricer pricer) : base(pricer)
        {
            CB = null;
            source = null;
            eta = pricer.eta;     //default rate: 1 for total default, 0 for partial default
            R = pricer.R;         //recovery factor
            X = pricer.X;         //recovery value: Face value or predefault value
            alpha = pricer.alpha; //parameter in hazard rate model
            S0 = pricer.S0;       //spot price
            p0 = pricer.p0;       //hazard rate
            p = pricer.p;         //hazard rate func
        }

        /// <summary>
        /// virtual copy constructor
        /// </summary>
        /// <returns></returns>
        protected override ConvertibleBondPDEPricer clone()
        {
            return new AFVPricer(this);
        }
    }
    /// <summary>
    /// 
    /// </summary>
    public class BoundaryCondition
    {
        /// <summary>
        /// 
        /// </summary>
        public enum BC_Type {
            /// <summary>
            /// 
            /// </summary>
            ZeroS,
            /// <summary>
            /// 
            /// </summary>
            LargeS,
            /// <summary>
            /// 
            /// </summary>
            Dirichlet };
        /// <summary>
        /// 
        /// </summary>
        public BC_Type bc_type = 0;
        /// <summary>
        /// 
        /// </summary>
        public double[] bc_values = new double[4];
        /// <summary>
        /// 
        /// </summary>
        public BoundaryCondition()
        {
        }
    }

    /// <summary>
    /// parameters for Black-Scholes equation
    /// </summary>
    public class BlkSch
    {
        /// <summary>
        /// 
        /// </summary>
        public double[] sol, x, source;
        /// <summary>
        /// 
        /// </summary>
        public Func<double,double,double> c2, c1, c0;
        /// <summary>
        /// 
        /// </summary>
        public BoundaryCondition[] bc;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sol"></param>
        /// <param name="x"></param>
        /// <param name="s2d2Vd2s"></param>
        /// <param name="sdVds"></param>
        /// <param name="V"></param>
        /// <param name="source"></param>
        /// <param name="bc"></param>
        public BlkSch(
                double[] sol,
                double[] x,
                Func<double,double,double> s2d2Vd2s,
                Func<double,double,double> sdVds,
                Func<double,double,double> V,
                double[] source,
                BoundaryCondition[] bc)
        {
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
    class BS_PSOR
    {
        //for fully coupled equations
        public BlkSch[] bs;
        public double NumIter;
        public double ErrorNorm;

        //for numerical scheme
        private double theta;   //0 for explicit, 1 for implicit, 0.5 for Crank Nicolson
        private double dt;
        private double min_dx;
        private double t = 0;
        //for linear solver
        Action<double[][], int> constrain;
        enum DIR { UPPER, LOWER };

        public enum Scheme { Explicit, Implicit, CrankNicolson };
        public BS_PSOR(
                BlkSch[] bs,
                Action<double[][], int> constrain,
                Scheme scheme = Scheme.CrankNicolson)
        {

            this.constrain = constrain;
            this.bs = bs;
            min_dx = 1000;
            double[] x = bs[0].x;
            for (int i = 0; i < x.Length - 1; ++i)
            {
                min_dx = Math.Min(min_dx, x[i + 1] - x[i]);
            }
            switch (scheme)
            {
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

        public void constructLinearSystem(BlkSch BS, SparseLinearSystem SPS)
        {
            BoundaryCondition[] bc = BS.bc;
            double[] rhs = SPS.b;
            double[] x = BS.x;
            double[] sol = BS.sol;
            SPS.solution = BS.sol;
            double[] source = BS.source;
            int n = BS.sol.Length;
            applyBoundaryCondition(BS, bc[0], SPS.lower_bc, rhs, x, DIR.LOWER);
            applyBoundaryCondition(BS, bc[1], SPS.upper_bc, rhs, x, DIR.UPPER);
            for (int i = 1; i < n - 1; ++i)
            {
                double c2 = BS.c2(t, x[i]);
                double c1 = BS.c1(t, x[i]);
                double c0 = BS.c0(t, x[i]);
                //center in space
                double alpha = 2 * c2 * x[i] * x[i] / ((x[i] - x[i - 1]) * (x[i + 1] - x[i - 1])) - c1 * x[i] / (x[i + 1] - x[i - 1]);
                double beta = 2 * c2 * x[i] * x[i] / ((x[i + 1] - x[i]) * (x[i + 1] - x[i - 1])) + c1 * x[i] / (x[i + 1] - x[i - 1]);

                //using upwind scheme insead of center in space to elminate oscillation near discontinuities
                if (c1 * x[i] > 0)
                {
                    alpha = 2 * c2 * x[i] * x[i] / ((x[i] - x[i - 1]) * (x[i + 1] - x[i - 1]));
                    beta = 2 * c2 * x[i] * x[i] / ((x[i + 1] - x[i]) * (x[i + 1] - x[i - 1])) + c1 * x[i] / (x[i + 1] - x[i]);
                }
                else
                {
                    alpha = 2 * c2 * x[i] * x[i] / ((x[i] - x[i - 1]) * (x[i + 1] - x[i - 1])) - c1 * x[i] / (x[i] - x[i - 1]);
                    beta = 2 * c2 * x[i] * x[i] / ((x[i + 1] - x[i]) * (x[i + 1] - x[i - 1]));
                }
                
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

        public void advance(double t, double dt)
        {
            this.dt = dt;
            this.t = t;
            SparseLinearSystem[] sps = new SparseLinearSystem[bs.Length];
            for (int i = 0; i < bs.Length; ++i)
            {
                sps[i] = new SparseLinearSystem(bs[i].sol.Length);
                constructLinearSystem(bs[i], sps[i]);
            }

            PSOR psor_solver = new PSOR(sps, 1.55, constrain);
            psor_solver.tolerance = 1e-5;
            psor_solver.maxIter = 400;
            psor_solver.solve();
            NumIter = psor_solver.NumIter;
            ErrorNorm = psor_solver.ErrorNorm;
        }

        void applyBoundaryCondition(BlkSch bs, BoundaryCondition bc, double[] coef, double[] rhs, double[] x, DIR dir)
        {
            switch (bc.bc_type)
            {
                case BoundaryCondition.BC_Type.ZeroS:
                    if (dir == DIR.UPPER)
                        throw new InvalidArgumentException("Cannot use zero boundary on upper side");
                    coef[0] = 1 - bs.c0(t,x[0]) * theta * dt;
                    rhs[0] = (1 + bs.c0(t,x[0]) * (1 - theta) * dt) * bs.sol[0];
                    if (bs.source != null)
                        rhs[0] += bs.source[0] * dt;
                    break;
                case BoundaryCondition.BC_Type.LargeS:
                    if (dir == DIR.LOWER)
                    {
                        throw new InvalidArgumentException("Cannot use large s boundary on lower side");
                    }
                    else
                    {
                        int m = x.Length - 1;
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
    }

    class SparseLinearSystem
    {
        public double[] L;
        public double[] R;
        public double[] D;
        public double[] b;
        public double[] lower_bc;
        public double[] upper_bc;
        public double[] solution;

        public SparseLinearSystem(int n)
        {
            L = new double[n - 2];
            D = new double[n - 2];
            R = new double[n - 2];
            b = new double[n];
            lower_bc = new double[n];
            upper_bc = new double[n];
        }

    }

    /// <summary>
    /// Projected successive over-relaxation method for solving Ax = b with constraints,
    /// A is nxn matrix, b is nx1 vector, x is nx1 vector
    /// This class is only designed for 1D problem,
    /// hence the inputs are tri-diagonal band matrix A, right-hand side b,
    /// and relaxiation factor, constraints on the solution
    /// </summary>
    class PSOR
    {
        private SparseLinearSystem[] sls;
        private double w;
        private Action<double[][], int> applyConstraint;
        private double[][] x_new;
        private double[][] x_old;

        public double tolerance = 1e-5;
        public int maxIter = 100;
        public double ErrorNorm { get; private set; }
        public int NumIter { get; private set; }

        public PSOR(SparseLinearSystem[] sparseLinearSystem, double RelaxFactor, Action<double[][], int> constraint)
        {
            this.sls = sparseLinearSystem;
            w = RelaxFactor;
            applyConstraint = constraint;
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
            for (int i = 0; i < sls.Length; ++i)
            {
                foreach (double di in sls[i].D)
                {
                    if (di == 0)
                        throw new ArgumentException("There are zeros in diagnal");
                }
                if (sls[i].lower_bc[0] == 0 || sls[i].upper_bc.Last() == 0)
                    throw new ArgumentException("There are zeros in diagnal");
            }
        }

        public void doOneIteration()
        {
            int n = x_new[0].Length;
            for (int sys_id = 0; sys_id < sls.Length; ++sys_id)
                Array.Copy(x_new[sys_id], x_old[sys_id], x_new[sys_id].Length);
            for (int i = 0; i < n; ++i)
            {
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
                        double tmp = 0;
                        for (int j = i + 1; j < n; ++j)
                            tmp += lower_bc[j] * x[j];
                        x[i] = (1 - w) * x[i]
                             + w / lower_bc[i] * (b[i] - tmp);
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
                if (applyConstraint != null)
                    applyConstraint.Invoke(x_new, i);
            }
        }

        public void solve()
        {
            NumIter = 0;
            ErrorNorm = 1000.0;
            while (ErrorNorm > tolerance && NumIter++ < maxIter)
            {
                doOneIteration();
                ErrorNorm = computeError(x_new, x_old);
                // Console.WriteLine("Iteration {0}, error norm = {1}", NumIter, ErrorNorm);
            }
            if (ErrorNorm > 1.0)
            {
                //Warning
                System.Diagnostics.Debug.WriteLine("Warning: iteration limit reached {1}, error L2-norm = {0}", ErrorNorm, maxIter);
                System.Diagnostics.Debug.WriteLine("Trying direct method, may take for a while");
                //safeguard: recompute solution using direct method
                double residue = NMathSolve(sls, applyConstraint);
                System.Diagnostics.Debug.WriteLine("Residue = {0}", residue);
            }
        }

        public double NMathSolve(SparseLinearSystem[] sls, Action<double[][], int> applyConstraint)
        {
            double residue = 0;
            //safeguard method: solving equation using LU decomposition, ineffective but robust
            for (int sys_id = 0; sys_id < sls.Length; ++sys_id)
            {
                double[] lower_bc = sls[sys_id].lower_bc;
                double[] upper_bc = sls[sys_id].upper_bc;
                double[] b = sls[sys_id].b;
                int msize = b.Length;
                double[] D = sls[sys_id].D;
                double[] L = sls[sys_id].L;
                double[] R = sls[sys_id].R;
                //assemble matrix and right-hand-side vector
                var BM = new DoubleBandMatrix(msize, msize, 4, 4);
                for (int i = 0; i < msize; ++i)
                {
                    if (i == 0)
                    {
                        for (int j = 0; j < lower_bc.Length; ++j)
                            if (lower_bc[j] != 0)
                                BM[i, j] = lower_bc[j];
                    }
                    else if (i == msize - 1)
                    {
                        for (int j = 0; j < upper_bc.Length; ++j)
                            if (upper_bc[j] != 0)
                                BM[i, j] = upper_bc[j];
                    }
                    else
                    {
                        BM[i, i] = D[i-1];
                        BM[i, i - 1] = L[i-1];
                        BM[i, i + 1] = R[i-1];
                    }
                }
                //solve using LU decomposition
                DoubleVector B = new DoubleVector(b);
                var LUDecomp = new DoubleBandFact(BM);
                var solution = LUDecomp.Solve(B).ToArray();
                Array.Copy(solution, x_new[sys_id], solution.Length);
                residue += computeResidue(sls[sys_id]);
            }

            if (applyConstraint != null)
                for (int i = 0; i < x_new[0].Length; ++i)
                    applyConstraint(x_new, i);
            return residue;
        }

        private double computeResidue(SparseLinearSystem sls)
        {
            //check residue
            int N = sls.solution.Length;
            double res = 0;
            double tmp = 0.0;
            for (int i = 0; i < N; ++i)
                tmp += sls.lower_bc[i] * sls.solution[i];
            res += Math.Abs(sls.b[0] - tmp);

            tmp = 0;
            for (int i = 0; i < N; ++i)
                tmp += sls.upper_bc[i] * sls.solution[i];
            res += Math.Abs(sls.b[N - 1] - tmp);

            for (int i = 1; i < N - 1; ++i)
            {
                tmp = sls.L[i - 1] * sls.solution[i - 1] + sls.D[i - 1] * sls.solution[i] + sls.R[i - 1] * sls.solution[i + 1];
                res += Math.Abs(sls.b[i] - tmp);
            }
            return res;
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
            System.Diagnostics.Debug.WriteLine(string.Join(",", array));
        }
    }

    class DescendingOrder : IComparer<double> {
        public int Compare(double a, double b) { return b.CompareTo(a); }
    }
}
