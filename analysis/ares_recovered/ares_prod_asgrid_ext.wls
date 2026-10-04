(* Exact NNLL cumulant and its alpha_s expansion for thrust on the EERAD3 tau grid,
   central scales plus the four factor-two variations of muR and x_V.
   refscale follows muR so aS is alpha_s(muR) and no symbolic running is needed.
   Columns: tau, muR0, X0, Sigma(as=0.118), c0, c1, c2, c3, alpha_s(muR), Sigma(alpha_s(muR)) *)
PrependTo[$Path, "./ares_recovered/ARES_sym"];
<<ARES`;
startT = AbsoluteTime[];
thr = BuildObservable["Thrust"];
unwrap[e_] := e /. Piecewise[{{x_, _}}, _] :> x;
taus = Table[0.00125 + 0.0025 k, {k, 0, 179}];
scales = {{1, 1}, {2, 1}, {1/2, 1}, {1, 2}, {1, 1/2}};
out = {};
Do[Module[{mu = sc[[1]], xv = sc[[2]]},
   (* alpha_s(muR) by the package's own running from alpha_s(MZ) = 0.118: the scale
      variation must move the coupling and the logarithms together *)
   asgrid = {0.100, 0.102, 0.104, 0.106, 0.108, 0.110, 0.112, 0.114, 0.116, 0.118, 0.120, 0.122, 0.124, 0.126, 0.128, 0.130, 0.132, 0.134, 0.136};
   asmu = N[AlphaSFixedNF[mu MZ, 3, {"refscale" -> MZ, "refalphas" -> 0.118}], 17];
   Print["  muR0=", N@mu, "  alpha_s(muR) = ", asmu];
   Do[Module[{sym, ser, c},
     sym = Resum[{}, Log[1/t], thr,
        {"Order" -> "NNLL", "refalphas" -> aS, "refscale" -> mu MZ, "Q" -> MZ,
         "muR0" -> mu, "X0" -> xv}];
     ser = unwrap[Normal[Series[sym, {aS, 0, 3}]]];
     c = Table[N[Coefficient[ser, aS, k], 17], {k, 0, 3}];
     AppendTo[out, Join[{t, N@mu, N@xv, N[unwrap[sym] /. aS -> 0.118, 17]}, c,
                        {asmu, N[unwrap[sym] /. aS -> asmu, 17]},
                        Table[N[unwrap[sym] /.
                          aS -> N[AlphaSFixedNF[mu MZ, 3,
                            {"refscale" -> MZ, "refalphas" -> a0}], 17], 17],
                          {a0, asgrid}]]]],
    {t, taus}];
   Print["scale muR0=", N@mu, " X0=", N@xv, "  elapsed ", Round[AbsoluteTime[] - startT], " s"]],
 {sc, scales}];
Export["./ares_recovered/ares_prod_asgrid_ext.csv", out];
Print["wrote ", Length[out], " rows"];
