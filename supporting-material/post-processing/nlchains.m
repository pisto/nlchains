(* ::Package:: *)

(* ::Input::Initialization:: *)
BeginPackage["nlchains`"];


(* ::Input::Initialization:: *)
KList::usage="KList[n] generates a list of wave numbers for a chain of length n, ordered as the result of a DFT";
LoadDump::usage="LoadDump[\"filename\", n] loads an ensemble of chains of length n for a real model (pairs of coordinates and momenta) from the specified file";
LoadComplexDump::usage="LoadComplexDump[\"filename\", n] loads an ensemble of chains of length n for a complex model (a list of complex values) from the specified file";
LoadEntropy::usage="LoadEntropy[\"filename\"] loads an entropy file with triplets containing the time stamp, the Wave Turbulence entropy and the information entropy";
LoadLinearEnergies::usage="LoadLinearEnergies[\"filename\"] loads a \"linenergies\" file with the linear energies of a model, returning a list of pairs of wave number and corresponing linear energy";
EnergyDNKG::usage="EnergyDNKG[\[Phi]\[Pi], m, \[Beta]] returns the total energy for a realization \[Phi]\[Pi] of the DNKG model, with mass parameter m and nonlinear parameter \[Beta]";
EnergyFPUT::usage="EnergyFPUT[\[Phi]\[Pi], \[Alpha], \[Beta]] returns the total energy for a realization \[Phi]\[Pi] of the FPUT model, with cubic and quartic nonlinear parameters \[Alpha] and \[Beta]";
EnergyToda::usage="EnergyToda[\[Phi]\[Pi], \[Alpha]] returns the total energy for a realization \[Phi]\[Pi] of the Toda model, with nonlinear parameter \[Alpha]";
EnergydDNKG::usage="EnergydDNKG[\[Phi]\[Pi], m] returns the total energy for a realization \[Phi]\[Pi] of the dDNKG model, with a list of mass parameters m and nonlinear parameter \[Beta]";
EnergyDNLS::usage="EnergyDNLS[\[Psi], \[Beta]] returns the total energy for a realization \[Psi] of the DNLS model, with nonlinear parameter \[Beta]";
EigendDNKG::usage="EigendDNKG[m] returns the eigensystem of the dDNKG model with a list of mass parameters m, as a list of pairs of pulsation (square root of eigenvalues) and the corresponding eigenvector";
EntropyINF::usage="EntropyINF[e] returns the information entropy for the list of linear energies e";
EntropyWT::usage="EntropyWT[e] returns the Wave Turbulence entropy for the list of linear energies e";


(* ::Input::Initialization:: *)
Begin["`Private`"];


(* ::Input::Initialization:: *)
KList[n_Integer]/;Positive@n&&EvenQ@n:=Join[Range[0,n/2-1],Range[-n/2,-1]]
KList[n_Integer]/;Positive@n&&OddQ@n:=Join[Range[0,(n-1)/2],Range[-(n-1)/2,-1]]


(* ::Input::Initialization:: *)
LoadDump[name_String,n_Integer]:=With[{l=BinaryReadList[name,{"Real64","Real64"}]},ArrayReshape[l,{Length@l/n,n,2}]]
LoadComplexDump[name_String,n_Integer]:=With[{l=BinaryReadList[name,"Complex128"]},ArrayReshape[l,{Length@l/n,n}]]
LoadEntropy[name_String]:=With[{l=BinaryReadList[name,"Real64"]},ArrayReshape[l,{Length@l/3,3}]]
LoadLinearEnergies[name_String]:=With[{l=BinaryReadList[name,"Real64"]},Transpose@{N@KList@Length@l,l}]


(* ::Input::Initialization:: *)
EnergyDNKG[\[Phi]pi_?ArrayQ,m_,g_]/;MatchQ[Dimensions@\[Phi]pi,{_,2}]:=With[{\[Phi]=\[Phi]pi[[All,1]],pi=\[Phi]pi[[All,2]]},
pi^2/2+ListConvolve[{1,-1},\[Phi],{-1,-1}]^2/2+m \[Phi]^2/2+g \[Phi]^4/4]//Total
EnergyFPUT[\[Phi]pi_?ArrayQ,\[Alpha]_,\[Beta]_]/;MatchQ[Dimensions@\[Phi]pi,{_,2}]:=With[{\[Phi]diff=ListConvolve[{1,-1},\[Phi]pi[[All,1]],{-1,-1}],pi=\[Phi]pi[[All,2]]},
pi^2/2+\[Phi]diff^2/2+\[Alpha] \[Phi]diff^3/3+\[Beta] \[Phi]diff^4/4]//Total
EnergyToda[\[Phi]pi_?ArrayQ,\[Alpha]_]/;MatchQ[Dimensions@\[Phi]pi,{_,2}]:=With[{\[Phi]diff=ListConvolve[{1,-1},\[Phi]pi[[All,1]],{-1,-1}],pi=\[Phi]pi[[All,2]]},
pi^2/2+Exp[2 \[Phi]diff \[Alpha]]/(4 \[Alpha]^2)-\[Phi]diff/(2 \[Alpha])-1/(4 \[Alpha]^2)]//Total
EnergyDNLS[a_?ArrayQ,g_]/;ArrayDepth@a==1:=Abs@ListConvolve[{1,-1},a,{-1,-1}]^2+g/2Abs@a^4//Total
EnergydDNKG[\[Phi]pi_?ArrayQ,m_?ArrayQ,g_]/;MatchQ[Dimensions@\[Phi]pi,{_,2}]&&Length@m==Length@\[Phi]pi:=With[{\[Phi]=\[Phi]pi[[All,1]],pi=\[Phi]pi[[All,2]]},
pi^2/2+ListConvolve[{1,-1},\[Phi],{-1,-1}]^2/2+m.\[Phi]^2/2+g \[Phi]^4/4]//Total


(* ::Input::Initialization:: *)
EigendDNKG[m_?ArrayQ]/;ArrayDepth@m==1:=Sort@Transpose@MapAt[Sqrt,Quiet@Eigensystem@SparseArray@{Band@{1,1}->2.+m,Band@{2,1}->-1.,Band@{1,2}->-1.,{1,-1}->-1.,{-1,1}->-1.},1]


(* ::Input::Initialization:: *)
EntropyINF[energies_?ArrayQ]/;ArrayDepth@energies==1:=With[{fk=Select[energies,Not@*PossibleZeroQ]Length@energies/Total@energies},Total[fk Log@fk]]
EntropyWT[energies_?ArrayQ]/;ArrayDepth@energies==1:=With[{fk=energies Length@energies/Total@energies},-Total[Log@fk]]


(* ::Input::Initialization:: *)
End[];


(* ::Input::Initialization:: *)
EndPackage[]
