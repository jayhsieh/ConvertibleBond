using Axioma.Phoenix.Platform.MathsLayer;
using Axioma.Phoenix.Platform.MathsLayer.SecurityPricingHelpers.PDEFramework;
using Axioma.Pricers.Callable;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace Axioma.Phoenix.Valuation.Framework.Tests
{
    [TestClass]
    public class PDEframeworkTest
    {

        private Services.IValuationFramework _framework = Services.ValuationFrameworkFactory.CreateValuationFramework();
        private Axioma.Phoenix.Valuation.Framework.Tests.ValuationDataFeed _feed = ValuationDataFeed.ValuationDataSingleton;
        //        //        [TestMethod, TestCategory(TestCategories.Valuation), TestCategory(TestCategories.Pricers_Axioma)]
        //        //        public void explicitTest()
        //        //        {
        //        //            double sigma = 0.3;
        //        //            double r = 0.03;
        //        //            double D = 0.0;
        //        //            double K = 50;
        //        //            double T = 1.0;
        //        //            double S = 60;
        //        //            double SMax = 5 * K;
        //        //            double[] sArray = new double[2] { 0.5, 1.1 };
        //        //            Grid G_T = new Grid1D(0, T, 0.0001);
        //        //            Grid G_X = new Grid1D(0, SMax, 1, sArray);

        //        //            // Coefficients of PDE.
        //        //            Func<double, double, double> Partial_x = (x, t) =>( r-D) * x;
        //        //            Func<double, double, double> Partial_xx = (x, t) => 0.5 * sigma * sigma * x * x;
        //        //            Func<double, double, double> V_x_t = (x, t) => -r;

        //        //            //Boundary Conditions  for x=0 and x= x_max for all t.
        //        //            Func<double, double> BCBottom = (t) => 0;
        //        //            Func<double, double> BCTop = (t) => SMax - K * Math.Exp(-r * t);
        //        //            BType bcTop = new Fixed();
        //        //            BType bcBottom = new Fixed();
        //        //            BConditions Cond = new BConditions(bcTop, bcBottom);

        //        //            //Initial condition for all x and t=0.
        //        //            Func<double, double> IC = (x) => Math.Max(x - K, 0);

        //        //            Solver expliciT = new Explicit(Partial_x, Partial_xx, V_x_t, BCBottom, BCTop, IC,Cond);
        //        //            PDEEngine pdeEngine = new PDEEngine(G_X, G_T, expliciT);
        //        //            pdeEngine.Evolution(T);
        //        //            double optionValue = pdeEngine.ValueGivenUnderlying(S);
        //        //        }

        //        //        [TestMethod, TestCategory(TestCategories.Valuation), TestCategory(TestCategories.Pricers_Axioma)]
        //        //        public void grid2Dtest()
        //        //        {
        //        //            double K = 50;
        //        //            double T = 1.0;
        //        //            double SMax = 5 * K;
        //        //            Grid G_T = new Grid1D(0, T, 0.0001);
        //        //            Grid G_X = new Grid1D(10, SMax, 1);
        //        //            double[] start = new double[] { 0, 10 };
        //        //            double[] end = new double[] { T, SMax };
        //        //            double[] steps = new double[] { .01, 1 };
        //        //            Grid G_xy = new Grid2D(start, end, steps);
        //        //            double[][] test2DGrid = (double[][])G_xy.CreateGridPoints();
        //        //        }

        //        //        [TestMethod, TestCategory(TestCategories.Valuation), TestCategory(TestCategories.Pricers_Axioma)]
        //        //        public void implicitTest()
        //        //        {
        //        //            double sigma = 0.3;
        //        //            double r = 0.03;
        //        //            double D = 0.0;
        //        //            double K = 50;
        //        //            double T = 1.0;
        //        //            double SMax = 5 * K;
        //        //            double S = 60;
        //        //            Grid G_T = new Grid1D(0, T, 0.01);
        //        //            Grid G_X = new Grid1D(0, SMax, 1);

        //        //            // Coefficients of PDE.
        //        //            Func<double, double, double> Partial_x = (x, t) => (r-D) * x;
        //        //            Func<double, double, double> Partial_xx = (x, t) => 0.5 * sigma * sigma * x * x;
        //        //            Func<double, double, double> V_x_t = (x, t) => -(r);

        //        //            //Boundary Conditions  for x=0 and x= x_max for all t.
        //        //            Func<double, double> BCBottom = (t) => 0;
        //        //            Func<double, double> BCTop = (t) => SMax - K * Math.Exp(-r * t);

        //        //            //Initial condition for all x and t=0.
        //        //            Func<double, double> IC = (x) => Math.Max(x - K, 0);

        //        //            BType bcTop = new Fixed();
        //        //            BType bcBottom = new Fixed();
        //        //            BConditions Cond = new BConditions(bcTop, bcBottom);

        //        //            Solver impliciT = new Implicit(Partial_x, Partial_xx, V_x_t, BCBottom, BCTop, IC,Cond);
        //        //            PDEEngine pdeEngine = new PDEEngine(G_X, G_T, impliciT);
        //        //            pdeEngine.Evolution(T);
        //        //            double optionValue = pdeEngine.ValueGivenUnderlying(S);

        //        //           // var tests = new List<Tuple<string, double, double, double>>();
        //        //           // tests.Add(Tuple.Create("European Call", 13.7385, optionValue, 1e-1));
        //        //           // tests.AssertAreEqual(); // now performing the tests...
        //        //        }

        //        [TestMethod, TestCategory(TestCategories.Valuation), TestCategory(TestCategories.Pricers_Axioma)]
        //        public void CNTest()
        //        {
        //            double sigma = 0.3;
        //            double r = 0.03;
        //            double D = 0.0;
        //            double K = 50;
        //            double T = 1.0;
        //            double SMax = 5 * K;
        //            double S = 60;
        //            double[] times = new double[5] { 0, .25, 0.5, .75, 1.0 };
        //            double[] underlyings = new double[10] { 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 };
        //            Grid G_T = new Grid1D(0, T, 0.01, times);
        //            Grid G_X = new Grid1D(0, SMax, 1, underlyings);

        //            // Coefficients of PDE.
        //            Func<double, double, double> Partial_x = (x, t) => (r - D) * x;
        //            Func<double, double, double> Partial_xx = (x, t) => 0.5 * sigma * sigma * x * x;
        //            Func<double, double, double> V_x_t = (x, t) => -(r);

        //            //Boundary Conditions  for x=0 and x= x_max for all t.
        //            Func<double, double> BCBottom = (t) => 0;
        //            Func<double, double> BCTop = (t) => SMax - K * Math.Exp(-r * t);

        //            //Initial condition for all x and t=0.
        //            Func<double, double> IC = (x) => Math.Max(x - K, 0);
        //            BType bcTop = new Fixed();
        //            BType bcBottom = new Fixed();
        //            BConditions Cond = new BConditions(bcTop, bcBottom);

        //            Solver CN = new CrankNicolson(Partial_x, Partial_xx, V_x_t, BCBottom, BCTop, IC, Cond);
        //            PDEEngine pdeEngine = new PDEEngine(G_X, G_T, CN);
        //            pdeEngine.Evolution(T);
        //            double optionValue = pdeEngine.ValueGivenUnderlying(S);
        //            //var profile = pdeEngine.ValueProfileAcrossTimes(underlyings,times);
        //            //var tests = new List<Tuple<string, double, double, double>>();
        //            //tests.Add(Tuple.Create("European Call",13.7437 , optionValue, 1e-2));
        //            //tests.AssertAreEqual(); // now performing the tests...
        //        }

        //        [TestMethod, TestCategory(TestCategories.Valuation), TestCategory(TestCategories.Pricers_Axioma)]
        //        public void grid1DTest()
        //        {
        //            double K = 50;
        //            double SMax = 5 * K;
        //            double Start = 0;
        //            double End = SMax;
        //            double point = 42;
        //            double[] ptsInputArray = new double[2] { -10,40 };
        //            double[] ptsInputArray2 = new double[2] { point, point + 54 };
        //            Grid G1 = new Grid1D(0, SMax, 20);
        //            Grid G2 = new Grid1D(0, SMax, 20, ptsInputArray);
        //            Grid G3 = new Grid1D(0, SMax, 20, ptsInputArray2);
        //            var ptsArray1 = G1.CreateGridPoints();
        //            var ptsArray2 = G2.CreateGridPoints();
        //            var ptsArray3 = G3.CreateGridPoints();
        //        }

        //        //        [TestMethod, TestCategory(TestCategories.Valuation), TestCategory(TestCategories.Pricers_Axioma)]
        //        //        public void HW1FactorPDE_Implicit()
        //        //        {
        //        //            double tMax = 1.0;
        //        //            double r = .03;
        //        //            double a = .01;
        //        //            double[] tArray = new double[10] { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0 };
        //        //            double maturity = tMax;
        //        //            double dt = 0.01;
        //        //            Grid T = new Grid1D(0, tMax, dt);
        //        //            ConstantCurve riskFree = new ConstantCurve(0.03);
        //        //            Func<double, double> f = (t) => ((t + dt) * riskFree.GetValue(t + dt) - t * riskFree.GetValue(t)) / dt;
        //        //            double[] xArray = new double[9] { -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04 };
        //        //            Grid X = new Grid1D(-1, 1, .001, xArray);

        //        //            // Neumann BC 
        //        //            BType bcTop = new Continuous();
        //        //            BType bcBottom = new Continuous();
        //        //            BConditions Cond = new BConditions(bcTop, bcBottom);

        //        //            Func<double, double> BCBottom = (t) => 0.0;
        //        //            Func<double, double> BCTop = (t) => 0.0;

        //        //            //Initial condition for all x and t=0.
        //        //            Func<double, double> IC = (x) => 1.0;

        //        //            var tests = new List<Tuple<string, double, double, double>>();
        //        //            double[] sigma = new double[4] { 0.01, 0.1, 0.2, 0.3 };
        //        //            List<List<double>> profile = new List<List<double>>();
        //        //            for (int i = 0; i < 4; ++i)
        //        //            {
        //        //                Func<double, double> alpha = (t) => f(t) + 0.5 * sigma[i] * sigma[i] / a / a * Math.Pow((1 - Math.Exp(-a * (t))), 2);
        //        //                // Coefficients of PDE.
        //        //                Func<double, double, double> Partial_x = (x, t) => -a * x;
        //        //                Func<double, double, double> Partial_xx = (x, t) => 0.5 * sigma[i] * sigma[i];
        //        //                Func<double, double, double> V_x_t = (x, t) => -(x + alpha(t));
        //        //                Solver IMPCT = new Implicit(Partial_x, Partial_xx, V_x_t, BCBottom, BCTop, IC,Cond);
        //        //                PDEEngine pdeEngine = new PDEEngine(X, T, IMPCT);
        //        //                pdeEngine.Evolution(maturity);
        //        //                double optionValue = pdeEngine.ValueGivenUnderlying(0) * 100;
        //        //                tests.Add(Tuple.Create("Sigma", Math.Exp(-r * maturity) * 100, optionValue, 1.0));
        //        //                if (i == 2)
        //        //                {
        //        //                    profile = pdeEngine.ValueProfileAcrossTimes(xArray, tArray);
        //        //                }
        //        //            }
        //        //            tests.AssertAreEqual(); // now performing the tests...
        //        //        }

        //        //        [TestMethod, TestCategory(TestCategories.Valuation), TestCategory(TestCategories.Pricers_Axioma)]
        //        //        public void HW1FactorPDE_CN()
        //        //        {
        //        //            double tMax = 1.0;
        //        //            double r = .03;
        //        //            double a = .001;
        //        //            double[] tArray = new double[10] { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0 };
        //        //            double maturity = 1.0;
        //        //            double dt = 0.01;
        //        //            Grid T = new Grid1D(0, tMax, dt);
        //        //            ConstantCurve riskFree = new ConstantCurve(0.03);
        //        //            Func<double, double> f = (t) => ((t + dt) * riskFree.GetValue(t + dt) - t * riskFree.GetValue(t)) / dt;
        //        //            double[] xArray = new double[9] { -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04 };
        //        //            Grid X = new Grid1D(-1.0, 1.0, .001, xArray);

        //        //            // Neumann BC 

        //        //            BType bcTop = new Continuous();
        //        //            BType bcBottom = new Continuous();
        //        //            BConditions Cond = new BConditions(bcTop, bcBottom);

        //        //            Func<double, double> BCBottom = (t) => 0.0;
        //        //            Func<double, double> BCTop = (t) => 0.0;

        //        //            //Initial condition for all x and t=0.
        //        //            Func<double, double> IC = (x) => 1.0;

        //        //            var tests = new List<Tuple<string, double, double, double>>();
        //        //            double[] sigma = new double[4] { 0.01, 0.1, 0.2, 0.3 };
        //        //            List<List<double>> profile = new List<List<double>>();
        //        //            for (int i = 0; i < 4; ++i)
        //        //            {
        //        //                Func<double, double> alpha = (t) => f(t) + 0.5 * sigma[i] * sigma[i] / a / a * Math.Pow((1 - Math.Exp(-a * (t))), 2);
        //        //                // Coefficients of PDE.
        //        //                Func<double, double, double> Partial_x = (x, t) => -a * x;
        //        //                Func<double, double, double> Partial_xx = (x, t) => 0.5 * sigma[i] * sigma[i];
        //        //                Func<double, double, double> V_x_t = (x, t) => -(x + alpha(t));
        //        //                Solver CN = new CrankNicolson(Partial_x, Partial_xx, V_x_t, BCBottom, BCTop, IC, Cond);
        //        //                PDEEngine pdeEngine = new PDEEngine(X, T, CN);
        //        //                pdeEngine.Evolution(maturity);
        //        //                double optionValue = pdeEngine.ValueGivenUnderlying(0) * 100;
        //        //                tests.Add(Tuple.Create("Sigma", Math.Exp(-r * maturity) * 100, optionValue, 1.0));
        //        //                if (i == 2)
        //        //                {
        //        //                    profile = pdeEngine.ValueProfileAcrossTimes(xArray, tArray);
        //        //                }
        //        //            }
        //        //            tests.AssertAreEqual(); // now performing the tests...
        //        //        }

        //        [TestMethod, TestCategory(TestCategories.Valuation), TestCategory(TestCategories.Pricers_Axioma)]
        //        public void ZCBondHWPDE_2()
        //        {
        //            double notional = 100;
        //            double k = 0.96;
        //            double bondmat = 1.0;
        //            double optmat = 0.3;
        //            double a = 0.001;
        //            double sigma = .01;
        //            Func<double, double> riskFree = (t) =>  0.03;
        //            ZeroCouponBondHWPDE zcbond = new ZeroCouponBondHWPDE(a, sigma, bondmat, riskFree);
        //            zcbond.CreateAndSolvePDE();
        //            var bondValue = zcbond.PV * notional;
        //            ZeroCouponBondOptionHWPDE zcOption = new ZeroCouponBondOptionHWPDE(k, optmat, zcbond);
        //            double value = zcOption.ZCBondOptionValue() * notional;
        //            //List<double> OptionValueList = new List<double>();
        //            //List<double> ZCValueList = new List<double>();
        //            //for (int i = 1; i < 10; ++i)
        //            //{
        //            //    ZeroCouponBondOptionHWPDE zcOption1 = new ZeroCouponBondOptionHWPDE(k, (double)i / 10.0, zcbond);
        //            //    OptionValueList.Add(zcOption1.ZCBondOptionValue() * notional);
        //            //}
        //            // double[] tArray = new double[14] { 0.01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 2, 5, 10 };
        //            // var ZCList = zcbond.TermStructureValues(tArray);
        //        }

        //        //        [TestMethod, TestCategory(TestCategories.Valuation), TestCategory(TestCategories.Pricers_Axioma)]
        //        //        public void BondPricerHWPDE()
        //        //        {
        //        //            double notional = 100;
        //        //            List<double> paytm = new List<double> { 0.15,0.4,0.65,0.9 };
        //        //            double freq = 0.25;
        //        //            double bondmat = 0.9;
        //        //            double a = 0.001;
        //        //            double sigma = .01;
        //        //            Func<double, double> riskFree = (t) => Math.Sqrt(t)*0.03;
        //        //            List<double> dirtyPV = new List<double>();
        //        //            List<double> cleanPV = new List<double>();
        //        //            List<double> accrued = new List<double>();
        //        //            for (int i=1;i<6;++i)
        //        //            {
        //        //                BondPricerHWPDE Bond = new BondPricerHWPDE(0.01*i, paytm, freq, a, sigma, bondmat, riskFree);
        //        //                Bond.CreateAndSolvePDE();
        //        //                dirtyPV.Add( Bond.DirtyPV * notional);
        //        //                cleanPV.Add(Bond.CleanPV * notional);
        //        //                accrued.Add( Bond.AccruedInterest * notional);
        //        //            }
        //        //        }

        //        //        [TestMethod, TestCategory(TestCategories.Valuation), TestCategory(TestCategories.Pricers_Axioma)]
        //        //        public void CallableBondPricerHWPDE()
        //        //        {
        //        //            double notional = 100;
        //        //            List<double> paytm = new List<double> { 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0 };
        //        //            double freq = 0.25;
        //        //            double bondmat = 2.0;
        //        //            double a = 0.001;
        //        //            double sigma = .01;
        //        //            double cpn = 0.03;
        //        //            List<double> exTimes = new List<double> { .75, 1.0, 1.25, 1.5, 1.75 };
        //        //            List<double> callPrices = new List<double> { 1, 1, 1, 1, 1 };
        //        //            List<double> putPrices = new List<double> { 1, 1, 1, 1, 1 };
        //        //            //double[] putPrices = new double[3] { 1.00, 1.01,1.02, };
        //        //            CallPutSchedule cpSchd = new CallPutSchedule(exTimes, callPrices,putPrices);
        //        //            List<double> rate = new List<double> { 0.01, .02, .03, .04, .05, .06, .07 };

        //        //            List<double> CdirtyPV = new List<double>();
        //        //            List<double> CcleanPV = new List<double>();
        //        //            List<double> Caccrued = new List<double>();
        //        //            List<double> dirtyPV = new List<double>();
        //        //            List<double> cleanPV = new List<double>();
        //        //            List<double> accrued = new List<double>();
        //        //            for (int i = 0; i < 7; ++i)
        //        //            {
        //        //                Func<double, double> riskFree = (t) => rate[i];
        //        //                CallableBondPricerHWPDE CallableBond = new CallableBondPricerHWPDE(cpn, paytm, freq, a, sigma, bondmat, riskFree, cpSchd);
        //        //                CallableBond.CreateAndSolvePDE();
        //        //                CdirtyPV.Add(CallableBond.DirtyPV * notional);
        //        //                CcleanPV.Add(CallableBond.CleanPV * notional);
        //        //                Caccrued.Add(CallableBond.AccruedInterest * notional);

        //        //                BondPricerHWPDE Bond = new BondPricerHWPDE(cpn, paytm, freq, a, sigma, bondmat, riskFree);
        //        //                Bond.CreateAndSolvePDE();
        //        //                dirtyPV.Add(Bond.DirtyPV * notional);
        //        //                cleanPV.Add(Bond.CleanPV * notional);
        //        //                accrued.Add(Bond.AccruedInterest * notional);
        //        //            }
        //        //        }

        //        [TestMethod, TestCategory(TestCategories.Valuation), TestCategory(TestCategories.Pricers_Axioma)]
        //        public void CallableBondPricerHWPDE2()
        //        {
        //            double notional = 1000;
        //            List<double> paytm = new List<double> { 0,0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0 };
        //            double freq = 0.25;
        //            double bondmat = 2.0;
        //            double a = 0.001;
        //            double sigma = .01;
        //            double cpn = 0.03;
        //            List<double> exTimes = new List<double> { .75, 1.0, 1.25, 1.5, 1.75 };
        //            List<double> callPrices = new List<double> { 1, 1, 1, 1, 1 };
        //            List<double> putPrices = new List<double> { 1, 1, 1, 1, 1 };
        //            CallPutSchedulePDE cpSchd = new CallPutSchedulePDE(exTimes, callPrices);
        //            List<double> rate = new List<double> { 0.01, .02, .03, .04, .05, .06, .07 };

        //            List<double> CdirtyPV = new List<double>();
        //            List<double> CcleanPV = new List<double>();
        //            List<double> Caccrued = new List<double>();
        //            List<double> dirtyPV = new List<double>();
        //            List<double> cleanPV = new List<double>();
        //            List<double> accrued = new List<double>();
        //            Func<double, double> riskFree = (t) => 0.03;
        //            CallableBondPricerHWPDE CallableBond = new CallableBondPricerHWPDE(cpn, paytm, freq, a, sigma, bondmat, riskFree, cpSchd);
        //            CallableBond.CreateAndSolvePDE();
        //            CdirtyPV.Add(CallableBond.DirtyPV * notional);
        //            CcleanPV.Add(CallableBond.CleanPV * notional);
        //            Caccrued.Add(CallableBond.AccruedInterest * notional);

        //            BondPricerHWPDE Bond = new BondPricerHWPDE(cpn, paytm, freq, a, sigma, bondmat, riskFree);
        //            Bond.CreateAndSolvePDE();
        //            dirtyPV.Add(Bond.DirtyPV * notional);
        //            cleanPV.Add(Bond.CleanPV * notional);
        //            accrued.Add(Bond.AccruedInterest * notional);
        //        }

        [TestMethod, TestCategory(TestCategories.Valuation), TestCategory(TestCategories.Pricers_Axioma)]
        public void EuropeanOption()
        {
            double sigma = 0.3;
            double D = 0.0;
            double K = 50;
            double T = 1.0;
            double SMax = 5 * K;
            double S = 60;
            Func<double, double> r = (t) => 0.03;
            EquityOptionPDE Call = new EuropeanEquityOptionPDE(S, K, T, D, r, sigma);
            //EquityOptionPDE Put = new EuropeanOption(S, K, T, D, r, sigma,false);
            double callValue = Call.GetValue();
            //double putValue = Put.GetValue();
            //PDEGreeks callGreeks = new PDEGreeks(Call);
            //PDEGreeks putGreeks = new PDEGreeks(Put);

            //double Cdelta = callGreeks.Delta(S);
            //double Cgamma = callGreeks.Gamma(S);
            //double Cvega = callGreeks.Vega();
            //double Ctheta = callGreeks.Theta();

            //double Pdelta = putGreeks.Delta(S);
            //double Pgamma = putGreeks.Gamma(S);
            //double Pvega = putGreeks.Vega();
            //double Ptheta = putGreeks.Theta();
        }
        [TestMethod, TestCategory(TestCategories.Valuation), TestCategory(TestCategories.Pricers_Axioma)]
        public void AmericanOption()
        {
            double sigma = 0.3;
            double D = 0.0;
            double K = 50;
            double T = 1.0;
            double S = 60;
            Func<double, double> r = (t) => 0.03;
            EquityOptionPDE Call = new AmericanEquityOptionPDE(S, K, T, D, r, sigma);
            EquityOptionPDE Put = new AmericanEquityOptionPDE(S, K, T, D, r, sigma, false);
            double callValue = Call.GetValue();
            double putValue = Put.GetValue();
            PDEGreeks callGreeks = new PDEGreeks(Call);
            PDEGreeks putGreeks = new PDEGreeks(Put);

            double Cdelta = callGreeks.Delta(S);
            double Cgamma = callGreeks.Gamma(S);
            double Cvega = callGreeks.Vega();
            double Ctheta = callGreeks.Theta();

            double Pdelta = putGreeks.Delta(S);
            double Pgamma = putGreeks.Gamma(S);
            double Pvega = putGreeks.Vega();
            double Ptheta = putGreeks.Theta();
        }
        //        [TestMethod, TestCategory(TestCategories.Valuation), TestCategory(TestCategories.Pricers_Axioma)]
        //        public void BarrierOption()
        //        {
        //            double sigma = 0.3;
        //            double D = 0.0;
        //            double K = 50;
        //            double barrier = 30;
        //            double T = 1.0;
        //            double SMax = 5 * K;
        //            double S = 50;
        //            Func<double, double> r = (t) => 0.03;
        //            //EquityOptionPDE downoutCall = new DownAndOutCall(barrier, S, K, T, D, r, sigma);
        //            //double downoutcallValue = downoutCall.GetValue();
        //            //EquityOptionPDE downinCall = new DownAndInCall(barrier, S, K, T, D, r, sigma);
        //            //double downincallValue = downinCall.GetValue();
        //            //EquityOptionPDE upoutCall = new UpAndOutCall(barrier, S, K, T, D, r, sigma);
        //            //double upoutcallValue = upoutCall.GetValue();
        //            //EquityOptionPDE upinCall = new UpAndInCall(barrier, S, K, T, D, r, sigma);
        //            //double upincallValue = upinCall.GetValue();
        //            EquityOptionPDE Put = new EuropeanEquityOptionPDE(S, K, T, D, r, sigma,false);
        //            double put = Put.GetValue();
        //            //double testIn = upincallValue + upoutcallValue;
        //            //double testout = downincallValue + downoutcallValue;
        //            //double dobarrier = downoutCall.pdeEngine.ValueGivenUnderlying(barrier);
        //            //double dibarrier = downinCall.pdeEngine.ValueGivenUnderlying(barrier);
        //            //double uobarrier = upoutCall.pdeEngine.ValueGivenUnderlying(barrier);
        //            //double uibarrier = upinCall.pdeEngine.ValueGivenUnderlying(barrier);
        //            //double callbarrier = Call.pdeEngine.ValueGivenUnderlying(barrier);
        //            //double[] underlying = new double[] {0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5,6,7,8,9,10,15,20,30,40};
        //            double[] underlying = new double[] { 0, 1, 2.5, 3.21, 4.45, 8, 10, 15, 20, 30, 40, 50, 60, 80, 90, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 110, 120, 130, 140, 150, 160, 170, 180, 200, 220, 250, 280, 300,330,350,380,400,450,480,490,500 };

        //            ////List<List<double>> upandinProfile = upinCall.pdeEngine.ValueProfileAcrossTimes(underlying, new double[] { 0, .1, .4, .6, .7, .8, 1.0 });
        //            ////List<List<double>> downandinProfile = downinCall.pdeEngine.ValueProfileAcrossTimes(underlying, new double[] { 0, .1, .4, .6, .7, .8, 1.0 });
        //            ////List<List<double>> upandoutProfile = upoutCall.pdeEngine.ValueProfileAcrossTimes(underlying, new double[] { 0, .1, .4, .6, .7, .8, 1.0 });

        //            //List<List<double>> downandoutProfile = downoutCall.pdeEngine.ValueProfileAcrossTimes(underlying, new double[] { 0, .1, .4, .6, .7, .8, 1.0 });
        //            PDEGreeks Greeks = new PDEGreeks(Put);
        //            //double Cdelta = Greeks.Delta(S);
        //            //double Cgamma = Greeks.Gamma(30);
        //            //double Cvega = callGreeks.Vega();
        //            //double Ctheta = callGreeks.Theta();
        //            List<double> Profile = Greeks.GammaProfile(underlying);

        //            //double Pdelta = putGreeks.Delta();
        //            //double Pgamma = putGreeks.Gamma();
        //            //double Pvega = putGreeks.Vega();
        //            //double Ptheta = putGreeks.Theta();
        //            // writing data to csv
        //            var file = @"D:\PDE FRAMEWORK\data.csv";
        //            if (!File.Exists(file))
        //            {
        //                File.Create(file).Close();
        //            }
        //            using (var stream = File.CreateText(file))
        //            {
        //                //foreach (var item in Profile)
        //                //{
        //                //    string first = item[0].ToString();
        //                //    string second = item[1].ToString();
        //                //    string third = item[2].ToString();
        //                //    string four = item[3].ToString();
        //                //    string five = item[4].ToString();
        //                //    string six = item[5].ToString();
        //                //    string seven = item[6].ToString();
        //                //    string eight = item[7].ToString();
        //                //    string nine = item[8].ToString();
        //                //    string ten = item[9].ToString();
        //                //    string eleven = item[10].ToString();
        //                //    string twelve = item[11].ToString();
        //                //    string thirt = item[12].ToString();
        //                //    string fourt = item[13].ToString();
        //                //    string fift = item[14].ToString();
        //                //    string sixt = item[15].ToString();
        //                //    string csvRow = string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{11},{12},{13},{14},{15}", first, second, third, four, five, six, seven, eight, nine, ten, eleven, twelve, thirt, fourt, fift, sixt);
        //                //    //string csvRow = string.Format("{0},{1},{2},{3}", seven, eight, nine, ten);
        //                //    stream.WriteLine(csvRow);
        //                //}
        //                foreach(var item in Profile)
        //                {
        //                    string value = item.ToString();
        //                    string csvRow = string.Format("{0}",value);
        //                    stream.WriteLine(csvRow);
        //                }
        //            }
        //            }

        //        [TestMethod, TestCategory(TestCategories.Valuation), TestCategory(TestCategories.Pricers_Axioma)]
        //        public void CallableBondPricerHWPDEVsAnalytics()
        //        {
        //            var issueDate = new DateTime(2011, 12, 1);
        //            var maturityDate = new DateTime(2020, 12, 1);
        //            DateTime pricingDate = new DateTime(2012, 4, 1);
        //            var riskFreeCurve = ValuationDataFeed.RefUSDRiskFreeCurve();
        //            DataBinder collector = new DataBinder();
        //            collector.RequireData(riskFreeCurve);
        //            var result = collector.BindData(_feed);
        //            var riskcurve = riskFreeCurve.Data.GetTermStructure(TimeMeasure.FractionOfYearACT365F, InterpolationMethod.PieceWiseLinear, FlatFilledScenario.ThreadSingleton.CreateCrossSection(pricingDate));
        //            Func<double, double> riskfunction = (x) => riskcurve.GetValue(x);

        //            var CallSch= new Schedule.FrequencyBased(new DateTime(2018, 12, 1), new DateTime(2020, 12, 1), Frequencies.SemiAnnual, true, true, true);
        //            var CpnSch = new Schedule.FrequencyBased(issueDate, maturityDate, Frequencies.SemiAnnual, true, true, true);

        //            Func<DateTime, double> fractionOfYearFromValuationDateFunc = (date) => pricingDate <= date
        //                                                                    ? TimeMeasure.FractionOfYearACT365F.Measure(pricingDate, date)
        //                                                                    : -TimeMeasure.FractionOfYearACT365F.Measure(date, pricingDate);
        //            DateTime firstCpnDateBeforeVal= new DateTime();
        //            foreach (var date in CpnSch)
        //            {
        //                if (date > pricingDate)
        //                {
        //                    firstCpnDateBeforeVal = CpnSch.AssociatedFrequency.PreviousDate(date);
        //                    break;
        //                }
        //            }
        //            double accruedLastCpn = fractionOfYearFromValuationDateFunc(firstCpnDateBeforeVal);
        //            double notional = 100;
        //            List<double> paytm = new List<double>();
        //            double freq = 0.5;
        //            double a = 0.001;
        //            double sigma = 0.0092485;
        //            double cpn = 0.05;
        //            var cpndateArray = CpnSch.UnadjustedDates.ToArray();
        //            int k = 0;
        //            while (cpndateArray[k] < maturityDate)
        //            {
        //                paytm.Add(fractionOfYearFromValuationDateFunc(cpndateArray[k]));
        //                ++k;
        //            }
        //            paytm.Add(fractionOfYearFromValuationDateFunc(CpnSch.EndDate));
        //            double bondmat = paytm.Last();
        //            List<double> exTimes = new List<double>();
        //            var calldateArray = CallSch.UnadjustedDates.ToArray();
        //            int i = 0;
        //            while (calldateArray[i]<CallSch.EndDate)
        //            {
        //                exTimes.Add(fractionOfYearFromValuationDateFunc(calldateArray[i]));
        //                ++i;
        //            }

        //            exTimes.Add(fractionOfYearFromValuationDateFunc(CallSch.EndDate));
        //            List<double> callPrices = new List<double>();
        //            int j = 0;
        //            while(j<exTimes.Count)
        //            {
        //                callPrices.Add(1);
        //                ++j;
        //            }
        //            CallPutSchedulePDE cpSchd = new CallPutSchedulePDE(exTimes, callPrices);
        //            List<double> CdirtyPV = new List<double>();
        //            List<double> CcleanPV = new List<double>();
        //            List<double> Caccrued = new List<double>();
        //            CallableBondPricerHWPDE CallableBond = new CallableBondPricerHWPDE(cpn, paytm, freq, a, sigma, bondmat, riskfunction, cpSchd);
        //            CallableBond.CreateAndSolvePDE();
        //            CdirtyPV.Add(CallableBond.DirtyPV * notional);
        //            CcleanPV.Add(CallableBond.CleanPV * notional);
        //            Caccrued.Add(CallableBond.AccruedInterest * notional);
        //            var tests = new List<Tuple<string, double, double, double>>();
        //            tests.Add(Tuple.Create("Dirty", 123.0867171, CdirtyPV[0], 1e-3));   // from AxR test 
        //            tests.Add(Tuple.Create("Clean", 121.4200505, CcleanPV[0], 1e-3));  
        //            tests.Add(Tuple.Create("Accrued", 1.666666667, Caccrued[0], 1e-3)); 
        //           // tests.AssertAreEqual(); // now performing the tests...
        //        }

        //        [TestMethod, TestCategory(TestCategories.Valuation), TestCategory(TestCategories.Pricers_Axioma)]
        //        public void CallableBondPricerHWPDEVsAxR()
        //        {
        //            var issueDate = new DateTime(2012, 10, 1);
        //            var maturityDate = new DateTime(2017, 4, 1);
        //            DateTime pricingDate = new DateTime(2012, 12, 1);
        //            var riskFreeCurve = ValuationDataFeed.RefUSDRiskFreeCurve();
        //            DataBinder collector = new DataBinder();
        //            collector.RequireData(riskFreeCurve);
        //            var result = collector.BindData(_feed);
        //            var riskcurve = riskFreeCurve.Data.GetTermStructure(TimeMeasure.FractionOfYearACT365F, InterpolationMethod.PieceWiseLinear, FlatFilledScenario.ThreadSingleton.CreateCrossSection(pricingDate));
        //            Func<double, double> riskfunction = (x) => riskcurve.GetValue(x);

        //            var CallSch = new Schedule.FrequencyBased(new DateTime(2018, 12, 1), new DateTime(2020, 12, 1), Frequencies.SemiAnnual, true, true, true);
        //            var CpnSch = new Schedule.FrequencyBased(issueDate, maturityDate, Frequencies.SemiAnnual, true, true, true);

        //            Func<DateTime, double> fractionOfYearFromValuationDateFunc = (date) => pricingDate <= date
        //                                                                    ? TimeMeasure.FractionOfYearACT365F.Measure(pricingDate, date)
        //                                                                    : -TimeMeasure.FractionOfYearACT365F.Measure(date, pricingDate);
        //            DateTime firstCpnDateBeforeVal = new DateTime();
        //            foreach (var date in CpnSch)
        //            {
        //                if (date > pricingDate)
        //                {
        //                    firstCpnDateBeforeVal = CpnSch.AssociatedFrequency.PreviousDate(date);
        //                    break;
        //                }
        //            }
        //            double accruedLastCpn = fractionOfYearFromValuationDateFunc(firstCpnDateBeforeVal);
        //            double notional = 1000;
        //            List<double> paytm = new List<double>();
        //            double freq = 0.5;
        //            double a = 0.00100789;
        //            double sigma = 0.00852859;
        //            double cpn = 0.08375000;
        //            var cpndateArray = CpnSch.UnadjustedDates.ToArray();
        //            int k = 0;
        //            while (cpndateArray[k] < maturityDate)
        //            {
        //                paytm.Add(fractionOfYearFromValuationDateFunc(cpndateArray[k]));
        //                ++k;
        //            }
        //            paytm.Add(fractionOfYearFromValuationDateFunc(CpnSch.EndDate));
        //            double bondmat = paytm.Last();
        //            List<double> callPrices = new List<double>();
        //            List<double> exTimes = new List<double>();
        //            double temp = 0.0821917808219178;
        //            while (temp < 0.32328767123287672)
        //            {
        //                exTimes.Add(temp);
        //                callPrices.Add(1.04188);
        //                temp += 1.0 / 360.0;
        //            }
        //            exTimes.Add(0.32328767123287672);
        //            callPrices.Add(1.04188);

        //            temp = 0.33150684931506852;
        //            while (temp < 1.2027397260273973)
        //            {
        //                exTimes.Add(temp);
        //                callPrices.Add(1.02792);
        //                temp += 1.0 / 360.0;
        //            }
        //            exTimes.Add(1.2027397260273973);
        //            callPrices.Add(1.02792);

        //            temp = 1.3315068493150686;
        //            while (temp < 2.2027397260273971)
        //            {
        //                exTimes.Add(temp);
        //                callPrices.Add(1.01396);
        //                temp += 1.0 / 360.0;
        //            }
        //            exTimes.Add(2.2027397260273971);
        //            callPrices.Add(1.01396);

        //            temp = 2.3315068493150686;
        //            while (temp < 4.2219178082191782)
        //            {
        //                exTimes.Add(temp);
        //                callPrices.Add(1.0);
        //                temp += 1.0 / 360.0;
        //            }
        //            exTimes.Add(4.2219178082191782);
        //            callPrices.Add(1.0);

        //            CallPutSchedulePDE cpSchd = new CallPutSchedulePDE(exTimes, callPrices);
        //            List<double> CdirtyPV = new List<double>();
        //            List<double> CcleanPV = new List<double>();
        //            List<double> Caccrued = new List<double>();
        //            CallableBondPricerHWPDE CallableBond = new CallableBondPricerHWPDE(cpn, paytm, freq, a, sigma, bondmat, riskfunction, cpSchd);
        //            CallableBond.CreateAndSolvePDE();
        //            CdirtyPV.Add(CallableBond.DirtyPV * notional);
        //            CcleanPV.Add(CallableBond.CleanPV * notional);
        //            Caccrued.Add(CallableBond.AccruedInterest * notional);
        //            var tests = new List<Tuple<string, double, double, double>>();
        //            tests.Add(Tuple.Create("Dirty", 1062.75196, CdirtyPV[0], 1.0));   // from AxR Dev. ISIN =  US35671DAS45
        //            tests.Add(Tuple.Create("Clean", 1048.793626, CcleanPV[0], 1.0));
        //            tests.Add(Tuple.Create("Accrued", 13.95833333, Caccrued[0], 1.0));
        //            tests.AssertAreEqual(); // now performing the tests...
        //        }

        [TestMethod, TestCategory(TestCategories.Valuation), TestCategory(TestCategories.Pricers_Axioma)]
        public void ConvertibleBondPricerTest()
        {
            //hardcoding parameters below, please change if necessary
            double sigma = 0.2;         // volatility
            double T = 5;               // time to expiry (years)
            double kappa = 1.0;         // conversion ratio
            double B = 100;             // principal

            Func<double, double> r = (t) => 0.05;       // risk-free interest rate
            Func<double, double> rg = (t) => 0.05;      // growth rate of stock
            Func<double, double> rc = (t) => 0.02;      // credit spread

            //call schedule
            OptionalityPeriodParameters[] callScheduleArray = new OptionalityPeriodParameters[1];
            callScheduleArray[0] = new OptionalityPeriodParameters(2, 5, 1.1);
            System.Collections.Generic.IEnumerable<OptionalityPeriodParameters> CallSchedule = callScheduleArray;

            //put scheule
            OptionalityPeriodParameters[] putScheduleArray = new OptionalityPeriodParameters[1];
            putScheduleArray[0] = new OptionalityPeriodParameters(3, 3, 1.05);
            System.Collections.Generic.IEnumerable<OptionalityPeriodParameters> PutSchedule = putScheduleArray;

            //coupon payments
            System.Collections.Generic.List<CouponParameters> Coupons = new System.Collections.Generic.List<CouponParameters>();
            for (int i = 0; i < 10; ++i)
                Coupons.Add(new CouponParameters(0, T, 0.5 + i * 0.5, 4/B));

            //callable bond parameter
            Pricers.Convertible.ConvertibleBondParameters cparams = new Pricers.Convertible.ConvertibleBondParameters();
            cparams.Maturity = T;
            cparams.FaceValue = B;
            cparams.CallSchedule = CallSchedule;
            cparams.PutSchedule = null;// PutSchedule;
            cparams.Coupons = Coupons;

            //measure excution time
            for (int it = 4; it <= 4; it++ )
            {
                ConvertibleBondPDEPricer CBSolver = new TFPricer(cparams, kappa, sigma, r, rg, rc);
                /*ConvertibleBondPDEPricer CBSolver = new AFVPricer(cparams, 
                                                                  spotPrice : B, 
                                                                  conversionRatio: kappa, 
                                                                  volatility: sigma, 
                                                                  riskFree: r, 
                                                                  growthRate: rg, 
                                                                  creditSpread: rc, 
                                                                  hazardRateFunc: null, 
                                                                  hazardRate: 0.02, 
                                                                  defaultRate: 1, 
                                                                  recoveryFactor: 0, 
                                                                  isDirtyPrice: false);*/
                CBSolver.print_time_interval = 0.1;
                CBSolver.OutDir = "D:";
                CBSolver.GridSize = it*200;
                System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
                CBSolver.CreateAndSolvePDE();
                stopwatch.Stop();
                System.Diagnostics.Debug.WriteLine("Time cost: {0} ms",
                                                    stopwatch.ElapsedMilliseconds);
                double ans = CBSolver.getPrice(B);
                System.Diagnostics.Debug.WriteLine("price(0, 100) = {0} at grid {1}",
                                                    ans, CBSolver.GridSize);
                using (var stream = new System.IO.StreamWriter("D:/timecost.txt", true))
                {
                    stream.WriteLine("{0} {1}", CBSolver.GridSize, stopwatch.ElapsedMilliseconds);
                }
                using (var stream = new System.IO.StreamWriter("D:/value.txt", true))
                {
                    stream.WriteLine("{0} {1}", CBSolver.GridSize, ans);
                }
            }
        }
    }
}
