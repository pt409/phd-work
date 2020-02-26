#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(*(* Potential functional form as described by Wadley et al 2001 *)*)


(* ::Input:: *)
(*\[Phi][r_]:=(A Exp[-\[Alpha](r/re-1)])/(1+(r/re-\[Kappa])^20)-(B Exp[-\[Beta](r/re-1)])/(1+(r/re-\[Lambda])^20)*)


(* ::Input:: *)
(*\[Rho]i[r_]:=(fe Exp[-\[Beta](r/re-1)])/(1+(r/re-\[Lambda])^20)*)


(* ::Input:: *)
(*F1[\[Rho]_]:=Subscript[Fn,0]+Sum[Subscript[Fn,i](\[Rho]/\[Rho]n-1)^i,{i,1,3}]*)
(*F2[\[Rho]_]:=Subscript[F,0]+Sum[Subscript[F,i](\[Rho]/\[Rho]e-1)^i,{i,1,3}]*)
(*F3[\[Rho]_]:=Fe(1-Log[(\[Rho]/\[Rho]s)^\[Eta]])(\[Rho]/\[Rho]s)^\[Eta]*)
(*\[Rho]n=17/20 \[Rho]e;*)
(*\[Rho]0=23/20 \[Rho]e;*)


(* ::Input:: *)
(*(* Need derivatives for use in atomicrex code *)*)
(*D[\[Phi][r],r]*)


(* ::Input:: *)
(*D[\[Rho]i[r],r]*)


(* ::Input:: *)
(*F1'[\[Rho]]*)


(* ::Input:: *)
(*F2'[\[Rho]]*)


(* ::Input:: *)
(*F3'[\[Rho]]*)


(* ::Input:: *)
(*(* Need to evaluate these where the splines meet *)*)
(*F1[0]*)


(* ::Input:: *)
(*F1[\[Rho]n]*)


(* ::Input:: *)
(*F2[\[Rho]n]*)


(* ::Input:: *)
(*F2[\[Rho]0]*)


(* ::Input:: *)
(*Simplify[F3[\[Rho]0]/.\[Rho]s->\[Rho]e]*)


(* ::Input:: *)
(*(* Can now develop a set of constraints by eliminating variables *)*)


(* ::Input:: *)
(*const1 = Solve[F1[0]==0,Subscript[Fn,3]][[1]]*)


(* ::Input:: *)
(*const2 = ToRules[F1[\[Rho]n]==F2[\[Rho]n]]*)


(* ::Input:: *)
(*const1 = const1/.const2*)


(* ::Input:: *)
(*const3 = Expand[Solve[F1'[\[Rho]n]==F2'[\[Rho]n],Subscript[Fn,1]][[1]]]*)


(* ::Input:: *)
(*const1 = const1/.const3*)


(* ::Input:: *)
(*const4 = Expand[Normal[Solve[F2[\[Rho]0]==(F3[\[Rho]0]/.\[Rho]s->\[Rho]e),Subscript[F,2],Reals]]][[1]]*)


(* ::Input:: *)
(*const1=const1/.const4*)
(*const2=const2/.const4*)
(*const3=const3/.const4*)


(* ::Input:: *)
(*const5=Solve[(F2'[\[Rho]0]==(F3'[\[Rho]0]/.\[Rho]s->\[Rho]e))/.const4,Subscript[F,3]][[1]]*)


(* ::Input:: *)
(*const4=const4/.const5*)


(* ::Input:: *)
(*const5 = Expand[Simplify[const5,Assumptions->Element[\[Eta],Reals]]]*)


(* ::Input:: *)
(*const4=Expand[Simplify[const4,Assumptions->Element[\[Eta],Reals]]]*)


(* ::Input:: *)
(*const1=Expand[Simplify[const1/.const4/.const5,Assumptions->Element[\[Eta],Reals]]]*)
(*const2=Expand[Simplify[const2/.const4/.const5,Assumptions->Element[\[Eta],Reals]]]*)
(*const3=Expand[Simplify[const3/.const4/.const5,Assumptions->Element[\[Eta],Reals]]]*)


(* ::Input:: *)
(*N[const1]*)
(*N[const2]*)
(*N[const3]*)
(*N[const4]*)
(*N[const5]*)


(* ::Input:: *)
(*(* This is the final set of constraints which are imposed *)*)


(* ::Input:: *)
(*constraints=Join[const1,const2,const3,const4,const5]*)


(* ::Input:: *)
(*(* Gradient of the embedding function should be zero at equilibrium density *)*)


(* ::Input:: *)
(*const6 = Solve[F2'[\[Rho]e]==0,Subscript[F,1]][[1]]*)


(* ::Input:: *)
(*constraints=constraints/.const6*)


(* ::Input:: *)
(*constraints=Join[constraints,const6]*)


(* ::Input:: *)
(*(* Have x2 additional constraints if the 2nd derivative of the splines are also matched. *)*)


(* ::Input:: *)
(*const7=Solve[F1''[\[Rho]n]==F2''[\[Rho]n],Subscript[Fn,2]][[1]]/.constraints*)


(* ::Input:: *)
(*const8=Solve[(F2''[\[Rho]0]==(F3''[\[Rho]0]/.\[Rho]s->\[Rho]e))/.constraints,Fe][[1]]*)


(* ::Input:: *)
(*const7=Expand[const7/.const8]/.constraints*)


(* ::Input:: *)
(*const8=const8/.constraints*)


(* ::Input:: *)
(*constraints=constraints/.const7/.const8*)


(* ::Input:: *)
(*constraints=Join[constraints,const7,const8]*)


(* ::Input:: *)
(*(* Can now produce a set of equations to use in atomicrex xml input file *)*)


(* ::Input:: *)
(*Simplify[F1[\[Rho]]/.constraints,Assumptions->Element[\[Eta],Reals]]/.Log[23/20]->N[Log[23/20]]*)


(* ::Input:: *)
(*Simplify[F2[\[Rho]]/.constraints,Assumptions->Element[\[Eta],Reals]]/.Log[23/20]->N[Log[23/20]]*)


(* ::Input:: *)
(*Simplify[F3[\[Rho]]/.constraints,Assumptions->Element[\[Eta],Reals]]/.Log[23/20]->N[Log[23/20]]*)


(* ::Input:: *)
(*Simplify[F1'[\[Rho]]/.constraints,Assumptions->Element[\[Eta],Reals]]/.Log[23/20]->N[Log[23/20]]*)


(* ::Input:: *)
(*Simplify[F2'[\[Rho]]/.constraints,Assumptions->Element[\[Eta],Reals]]/.Log[23/20]->N[Log[23/20]]*)


(* ::Input:: *)
(*Simplify[F3'[\[Rho]]/.constraints,Assumptions->Element[\[Eta],Reals]]/.Log[23/20]->N[Log[23/20]]*)


(* ::Input:: *)
(*(* Actually need to implement cutoff separately *)*)
(*screening[r_] := 1/(1+(r/re-\[Kappa])^20)*)


(* ::Input:: *)
(*screening'[r]*)


(* ::Input:: *)
(*(* This is a screening with a "correct" functional form such that screening[rc]=0*)


(* ::Input:: *)
(*screening2[r_]:=1-1/(1+((r-rc)/h)^m)*)


(* ::Input:: *)
(*Plot[{screening[r]/.{\[Kappa]->0.5,re->2.739521},screening2[r]/.{rc->13.697605/2,h->0.5,m->4}},{r,0,10.0}]*)


(* ::Input:: *)
(*screeningSols=DeleteDuplicates[Assuming[{{\[Kappa],re}\[Element]Reals,\[Kappa]>0,re>0},Simplify[Solve[screening''[r]==0,r]]]]*)


(* ::Input:: *)
(*Plot[{screening[r]/.{\[Kappa]->0.5,re->2.739521},screening''[r]/.{\[Kappa]->0.5,re->2.739521}},{r,0,10.0}]*)


(* ::Input:: *)
(*N[screeningSols/.{\[Kappa]->0.5,re->2.739521}]*)


(* ::Input:: *)
(*midCutoff=r/.screeningSols[[5]]*)


(* ::Input:: *)
(*Plot[{screening[r]/.{\[Kappa]->0.5,re->2.739521},screening2[r]/.{rc->midCutoff/.{\[Kappa]->0.5,re->2.739521},h->0.5}},{r,0,10.0}]*)


(* ::Input:: *)
(*(* Repeat this for the other functional form of the screening potential *)*)
(*screeningSols2=DeleteDuplicates[Assuming[{{\[Kappa],re,m}\[Element]Reals,m\[Element]Integers,rc>0,h>0,m>2},Simplify[Solve[screening2''[r]==0,r]]]]*)


(* ::Input:: *)
(*N[screeningSols2/.{rc->midCutoff/.{\[Kappa]->0.5,re->2.739521},h->0.5,m->4}]*)


(* ::Input:: *)
(*midCutoff2=rc-h ((m-1)/(m+1))^(1/m)*)


(* ::Input:: *)
(*(* Will now match the original screening function to the better function *)*)
(*N[FullSimplify[Series[screening[r],{r,midCutoff,1}]]]*)


(* ::Input:: *)
(*Series[screening2[r],{r,midCutoff2,1}]*)


(* ::Input:: *)
(*(* Frst match the mid point of the slope*)*)
(*midCutoff==midCutoff2 ;*)
(*(* Then match the 1st order approx at this point *)*)
(*SeriesCoefficient[screening[r],{r,midCutoff,0}]==SeriesCoefficient[screening2[r],{r,midCutoff2,0}]; *)
(*SeriesCoefficient[screening[r],{r,midCutoff,1}]==SeriesCoefficient[screening2[r],{r,midCutoff2,1}] ;*)


(* ::Input:: *)
(*intercept2[m_]:=SeriesCoefficient[screening2[r],{r,midCutoff2,0}]*)
(*DiscretePlot[intercept2[m],{m,2,20,2}]*)


(* ::Input:: *)
(*(* This has a limiting value of a half for large m, hence choose m=20 *)*)


(* ::Input:: *)
(*rcTrue=rc/.Rationalize[Solve[midCutoff==N[midCutoff2/.m->6],rc]]*)


(* ::Input:: *)
(*Plot[{screening[r]/.{\[Kappa]->1.0,re->2.739521},screening2[r]/.{rc->rcTrue/.{\[Kappa]->1.0,re->2.739521}}/.h->0.7/.m->6},{r,0,10.0}]*)


(* ::Input:: *)
(*ClearAll[cutoffValue]*)


(* ::Input:: *)
(*(* Use this function to compute a value of the cutoff to be used *)*)
(*cutoffValue[\[Kappa]Old_,reOld_]:=N[rc/.Rationalize[Solve[midCutoff==N[midCutoff2/.m->4],rc]]/.{h->3.0,\[Kappa]->\[Kappa]Old,re->reOld}][[1]]*)


(* ::Input:: *)
(*cutoffValue[0.5,2.6504034015]*)


(* ::Input:: *)
(*(* The actual params used for two fitted potentials (Ru, Re) and one pre-existing one (Mo). *)*)


(* ::Input:: *)
(*paramsRe=Join[N[constraints/.{Subscript[F,0]->-2.49901,\[Eta]->-0.462922}],{Subscript[F,0]->-2.49901,\[Eta]->-0.462922},{A->1.99411,B->2.54106,\[Alpha]->7.75245,\[Beta]->2.54106,fe->2.601233487,re->2.739520961,\[Rho]e->90.324416,\[Rho]s->90.324416,\[Kappa]->0.5,\[Lambda]->1.0}];*)
(*paramsMo={re->2.728100,fe->2.723710,\[Rho]e->29.354065,\[Rho]s->29.354065,\[Alpha]->8.393531,\[Beta]->4.476550,A->0.708787,B->1.120373,\[Kappa]->0.137640,\[Lambda]->0.275280,Subscript[Fn,0]->-2.534992,Subscript[Fn,1]->-0.059605,Subscript[Fn,2]->0.193065,Subscript[Fn,3]->-2.282322,Subscript[F,0]->-2.54,Subscript[F,1]->0.0,Subscript[F,2]->0.200269,Subscript[F,3]->-0.148770,\[Eta]->0.391750,Fe->-2.539945};*)


(* ::Input:: *)
(*(* Some nice plots to compare the three potentials *)*)


(* ::Input:: *)
(*Plot[{\[Phi][r]/.paramsRe,\[Phi][r]/.paramsMo},{r,2.0,6.5},*)
(*AxesLabel->{r,"\!\(\*SubscriptBox[\(\[Phi]\), \(AA\)]\)(r) (eV)"},*)
(*PlotLabels->Placed[{"Re","Mo"},Below],*)
(*PlotStyle->{"",Dashed}]*)


(* ::Input:: *)
(*Plot[{\[Rho]i[r]/.paramsRe,\[Rho]i[r]/.paramsMo},{r,0.0,3.5},*)
(*AxesLabel->{r,"f(r)"},*)
(*PlotLabels->{Placed["Re",Above],Placed["Mo",Above]},*)
(*PlotStyle->{"",Dashed},*)
(*PlotRange->Full]*)


(* ::Input:: *)
(*F[\[Rho]_]:=HeavisideTheta[\[Rho]n-\[Rho]]F1[\[Rho]]+HeavisideTheta[\[Rho]-\[Rho]n]HeavisideTheta[\[Rho]0-\[Rho]]F2[\[Rho]]+HeavisideTheta[\[Rho]-\[Rho]0]F3[\[Rho]]*)


(* ::Input:: *)
(*Plot[{F[\[Rho]]/.paramsRe,F[\[Rho]]/.paramsMo},{\[Rho],0.0,200},*)
(*AxesLabel->{\[Rho],"F(\[Rho]) (eV)"},*)
(*PlotLabels->Placed[{"Re","Mo"},Below],*)
(*PlotStyle->{"",Dashed}]*)
