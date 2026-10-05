PrependTo[$Path, "./ares_recovered/ARES_sym"];
<<ARES`;
thr = BuildObservable["Thrust"];
unwrap[e_] := e /. Piecewise[{{x_, _}}, _] :> x;
Do[Module[{mu = sc[[1]], xv = sc[[2]], t0, sym, ser},
   t0 = AbsoluteTime[];
   sym = Resum[{}, Log[1/0.05], thr,
      {"Order"->"NNLL","refalphas"->aS,"refscale"->mu MZ,"Q"->MZ,"muR0"->mu,"X0"->xv}];
   ser = unwrap[Normal[Series[sym, {aS, 0, 3}]]];
   Print["SMOKE muR0=", N@mu, " X0=", N@xv, " secs=", Round[AbsoluteTime[]-t0, 0.1], " c1c2c3=",
         N[Table[Coefficient[ser, aS, k], {k, 1, 3}], 10]]],
 {sc, {{1,1},{2,1},{1,2}}}];
