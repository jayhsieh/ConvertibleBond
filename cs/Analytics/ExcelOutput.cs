using Axioma.Phoenix.Platform.MathsLayer.SecurityPricingHelpers.PDEFramework;
using Axioma.Phoenix.Valuation.Framework.Tests;
using Axioma.Pricers.Callable;
using Axioma.Pricers.Convertible;
using ExcelDna.Integration;
using System;
using System.Linq;

namespace ExcelDNA
{
    public class ExcelOutput
    {
        private static ConvertibleBondPDEPricer CBSolver = null;
        private static double _sigma, _T, _kappa, _B, _risk_free, _credit_spread, _stock_growth;
        private static double[,] _couponInfoMatrix, _callInfoMatrix, _putInfoMatrix;

        [ExcelFunction(Description = "Compute Greeks Gamma")]
        public static double getGamma(
            double query_price,
            double query_time,
            double sigma,
            double T,
            double kappa,
            double B,
            double risk_free,
            double credit_spread,
            double stock_growth,
            double[,] couponInfoMatrix,
            double[,] callInfoMatrix,
            double[,] putInfoMatrix)
        {
            if (CBSolver == null || isParametersChanged(sigma, T, kappa, B, risk_free,
        credit_spread, stock_growth, couponInfoMatrix, callInfoMatrix, putInfoMatrix))
            {
                System.Diagnostics.Debug.WriteLine("Parameters changed, rebuilt the pricer");
                //when first time or parameters changed, rebuild the pricer
                ExcelConvertibleBondPricer(sigma, T, kappa, B, risk_free,
                    credit_spread, stock_growth, couponInfoMatrix, callInfoMatrix, putInfoMatrix);
                _sigma = sigma;
                _T = T;
                _kappa = kappa;
                _B = B;
                _risk_free = risk_free;
                _credit_spread = credit_spread;
                _stock_growth = stock_growth;
                _couponInfoMatrix = couponInfoMatrix;
                _callInfoMatrix = callInfoMatrix;
                _putInfoMatrix = putInfoMatrix;
            }
            return CBSolver.computeGamma(query_price, query_time);
        }

        [ExcelFunction(Description = "Compute Greeks Delta")]
        public static double getDelta(
            double query_price, 
            double query_time,
            double sigma,
            double T,
            double kappa,
            double B,
            double risk_free,
            double credit_spread,
            double stock_growth,
            double[,] couponInfoMatrix,
            double[,] callInfoMatrix,
            double[,] putInfoMatrix)
        {
            if (CBSolver == null || isParametersChanged(sigma, T, kappa, B, risk_free,
        credit_spread, stock_growth, couponInfoMatrix, callInfoMatrix, putInfoMatrix))
            {
                System.Diagnostics.Debug.WriteLine("Parameters changed, rebuilt the pricer");
                //when first time or parameters changed, rebuild the pricer
                ExcelConvertibleBondPricer(sigma, T, kappa, B, risk_free,
                    credit_spread, stock_growth, couponInfoMatrix, callInfoMatrix, putInfoMatrix);
                _sigma = sigma;
                _T = T;
                _kappa = kappa;
                _B = B;
                _risk_free = risk_free;
                _credit_spread = credit_spread;
                _stock_growth = stock_growth;
                _couponInfoMatrix = couponInfoMatrix;
                _callInfoMatrix = callInfoMatrix;
                _putInfoMatrix = putInfoMatrix;
            }
            return CBSolver.computeDelta(query_price, query_time);
        }

        [ExcelFunction(Description = "Compute Greeks Vega")]
        public static double getVega(
            double query_price,
            double query_time,
            double sigma,
            double T,
            double kappa,
            double B,
            double risk_free,
            double credit_spread,
            double stock_growth,
            double[,] couponInfoMatrix,
            double[,] callInfoMatrix,
            double[,] putInfoMatrix)
        {
            if (CBSolver == null || isParametersChanged(sigma, T, kappa, B, risk_free,
                credit_spread, stock_growth, couponInfoMatrix, callInfoMatrix, putInfoMatrix))
            {
                System.Diagnostics.Debug.WriteLine("Parameters changed, rebuilt the pricer");
                //when first time or parameters changed, rebuild the pricer
                ExcelConvertibleBondPricer(sigma, T, kappa, B, risk_free,
                    credit_spread, stock_growth, couponInfoMatrix, callInfoMatrix, putInfoMatrix);
                _sigma = sigma;
                _T = T;
                _kappa = kappa;
                _B = B;
                _risk_free = risk_free;
                _credit_spread = credit_spread;
                _stock_growth = stock_growth;
                _couponInfoMatrix = couponInfoMatrix;
                _callInfoMatrix = callInfoMatrix;
                _putInfoMatrix = putInfoMatrix;
            }
            
            return CBSolver.computeVega(query_price, query_time);
        }

        [ExcelFunction(Description = "Get convertible bond price at Price S at Time t")]
        public static double getPrice(
            double query_price,
            double query_time,
            double sigma,
            double T,
            double kappa,
            double B,
            double risk_free,
            double credit_spread,
            double stock_growth,
            double[,] couponInfoMatrix,
            double[,] callInfoMatrix,
            double[,] putInfoMatrix)
        {
            if (CBSolver == null || isParametersChanged(sigma, T, kappa, B, risk_free,
                    credit_spread, stock_growth, couponInfoMatrix, callInfoMatrix, putInfoMatrix))
            {
                System.Diagnostics.Debug.WriteLine("Parameters changed, rebuilt the pricer");
                //when first time or parameters changed, rebuild the pricer
                ExcelConvertibleBondPricer(sigma, T, kappa, B, risk_free, 
                    credit_spread, stock_growth, couponInfoMatrix, callInfoMatrix, putInfoMatrix);
                _sigma = sigma;
                _T = T;
                _kappa = kappa;
                _B = B;
                _risk_free = risk_free;
                _credit_spread = credit_spread;
                _stock_growth = stock_growth;
                _couponInfoMatrix = couponInfoMatrix;
                _callInfoMatrix = callInfoMatrix;
                _putInfoMatrix = putInfoMatrix;
            }
            return CBSolver.getPrice(query_price, query_time);
        }

        private static bool isParametersChanged(
            double sigma, 
            double T, 
            double kappa, 
            double B, 
            double risk_free,
            double credit_spread, 
            double stock_growth, 
            double[,] couponInfoMatrix, 
            double[,] callInfoMatrix, 
            double[,] putInfoMatrix)
        {
            if (_sigma != sigma ||
                _T != T ||
                _kappa != kappa ||
                _B != B ||
                _risk_free != risk_free ||
                _credit_spread != credit_spread ||
                _stock_growth != stock_growth ||
                _couponInfoMatrix.Length != couponInfoMatrix.Length ||
                _callInfoMatrix.Length != callInfoMatrix.Length ||
                _putInfoMatrix.Length != putInfoMatrix.Length)
                return true;
            if (!couponInfoMatrix.Cast<double>().SequenceEqual(_couponInfoMatrix.Cast<double>()) ||
                !callInfoMatrix.Cast<double>().SequenceEqual(_callInfoMatrix.Cast<double>()) ||
                !putInfoMatrix.Cast <double>().SequenceEqual(_putInfoMatrix.Cast <double>()))
                return true;
            return false;
        }
        private static void ExcelConvertibleBondPricer(
            double sigma, 
            double T, 
            double kappa, 
            double B,
            double risk_free,
            double credit_spread,
            double stock_growth,
            double[,] couponInfoMatrix,
            double[,] callInfoMatrix,
            double[,] putInfoMatrix)
        {
            Func<double, double> r = (t) => risk_free;       // risk-free interest rate
            Func<double, double> rg = (t) => stock_growth;      // growth rate of stock
            Func<double, double> rc = (t) => credit_spread;      // credit spread

            //call schedule
            if (callInfoMatrix.GetLength(1) != 3)
                throw new ArgumentException("Call information should have 3 columns");
            OptionalityPeriodParameters[] callScheduleArray = new OptionalityPeriodParameters[callInfoMatrix.GetLength(0)];
            for (int i = 0; i < callInfoMatrix.GetLength(0); ++i)
                callScheduleArray[i] = new OptionalityPeriodParameters(callInfoMatrix[i,0], callInfoMatrix[i,1], callInfoMatrix[i,2]);
            System.Collections.Generic.IEnumerable<OptionalityPeriodParameters> CallSchedule = callScheduleArray;

            //put scheule
            if (putInfoMatrix.GetLength(1) != 3)
                throw new ArgumentException("Put information should have 3 columns");
            OptionalityPeriodParameters[] putScheduleArray = new OptionalityPeriodParameters[putInfoMatrix.GetLength(0)];
            for (int i = 0; i < putInfoMatrix.GetLength(0); ++i)
                putScheduleArray[i] = new OptionalityPeriodParameters(putInfoMatrix[i,0], putInfoMatrix[i,1], putInfoMatrix[i,2]);
            System.Collections.Generic.IEnumerable<OptionalityPeriodParameters> PutSchedule = putScheduleArray;

            //coupon payments
            System.Collections.Generic.List<CouponParameters> Coupons = new System.Collections.Generic.List<CouponParameters>();
            for (int i = 0; i < couponInfoMatrix.GetLength(0); ++i)
                Coupons.Add(new CouponParameters(0, T, couponInfoMatrix[i,0], couponInfoMatrix[i,1]));

            //callable bond parameter
            ConvertibleBondParameters cparams = new ConvertibleBondParameters();
            cparams.Maturity = T;
            cparams.FaceValue = B;
            cparams.CallSchedule = CallSchedule;
            cparams.PutSchedule = PutSchedule;
            cparams.Coupons = Coupons;

            CBSolver = new TFPricer(cparams, kappa, sigma, rc, r, rg);
            CBSolver.print_time_interval = 0.25;
            CBSolver.OutDir = "D:";
            CBSolver.GridSize = 300;
            System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
            CBSolver.CreateAndSolvePDE();
            stopwatch.Stop();
            System.Diagnostics.Debug.WriteLine("Time cost: {0} ms",
                                                stopwatch.ElapsedMilliseconds);
            using (var stream = new System.IO.StreamWriter("D:/timecost.txt", true))
            {
                stream.WriteLine("{0} {1}", CBSolver.GridSize, stopwatch.ElapsedMilliseconds);
            }
        }
    }
}
