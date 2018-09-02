(* ::Package:: *)

(*
 * MathematicaSVM - A hands-on introduction to Support Vector Machines using Mathematica (c)
 *
 * Copyright (c) 2015 Marco Fornoni <marco.fornoni@alumni.epfl.ch>
 *
 * This file is part of the MathematicaSVM Software.
 *
 * MathematicaSVM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * MathematicaSVM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MathematicaSVM. If not, see <http://www.gnu.org/licenses/>.
 *
 *  Created on: Feb 03, 2014
 *      Author: Marco Fornoni
*)

BeginPackage ["SVM`", {"Global`"}];

Unprotect[
	plotSize, loadData, createData, getTrTeData, hinge, err,
	linearKernel, gaussianKernel, computeGaussianKernel, estimateSigmaSQ, computeDist,
	trainMaxMargin, trainSoftMargin, trainSoftMarginHinge, trainZeroOneError,
	trainSoftMarginHingeWbias, trainZeroOneErrorWbias,
	trainHardMarginSVM, train1NormSoftMarginSVM, train2NormSoftMarginSVM,
	testMaxMargin, testLinearClassifier, testNonLinearSVM,
	plotData, plotLinearResults, plotKernelResults,
	runNoBiasExperiment, runMaxMarginExperiment, runSVMExperiment
];

SVM::usage = "SVM is a package that contains implementations of max-margin classifiers 
and Support Vector Machines, exploiting the Mathematica built-in Quadratic Programming Solver.
It also contain utilities to load or directly draw 2D datasets.";

xPos::usage = "Positive Features created by createData";
xNeg::usage = "Negative Features created by createData";
plotSize::usage"Sets the size in pixels of the produced plots";

loadData::usage = "{fTr, yTr, fTe, yTe}=loadData[filename] loads the dataset in filename";
createData::usage = "createData[] creates drawable a plot, to draw a 2D dataset";
getTrTeData::usage = "{fTr, yTr, fTe, yTe}=getTrTeData[trPerc] generates training and testing data, with the points 
plotted using createData, where trPerc specifying the training percentage";

hinge::usage = "hinge_loss=hinge[margin] computes the hinge-loss for a given margin";
err::usage = "zo_err=err[margin] computes the 0/1 error for a given margin";

linearKernel::usage = "K=linearKernel[fTr, fTe] function to computes the Linear Kernel between fTr and fTe..";
gaussianKernel::usage = "f=gaussianKernel[input], function that return a gaussianKernel function using 
the provided input. If input is a matrix, it uses it to estimate the sigma, otherwise sigma=input.";
computeGaussianKernel::usage = "K=computeGaussianKernel[fTr, fTe, sigmaSQ] computes the Gaussian kernel matrix between fTr and fTe, using sigmaSQ as the kernel parameter";
estimateSigmaSQ::usage = "sigmaSQ=estimateSigmaSQ[fTr] estimates the data variance of the data using fTr";
computeDist::usage = "D=computeDist[fTr, fTe] computes the pairwise distances between fTr and fTe";

trainMaxMargin::usage = "{model, margin}=trainMaxMargin[feats, labels] trains a linear max-margin classifier.";
trainSoftMargin::usage = "{model, margin}=trainSoftMarginHinge[feats, labels, c] trains a soft-margin linear classifier, 
with regularization parameter c";
trainSoftMarginHinge::usage = "{model, margin}=trainSoftMarginHinge[feats, labels, c] trains a hinge-loss based 
soft-margin linear classifier, with regularization parameter c";
trainZeroOneError::usage = "{model, margin}=trainZeroOneError[feats, labels, c, minimizer] trains a 0/1 loss classifier, 
with regularization parameter c and using 'minimizer' as an optimization function";

trainSoftMarginHingeWbias::usage = "{model, margin}=trainSoftMarginHingeWbias[feats, labels, lambda, minimizer] 
trains a hinge-loss based soft-margin linear classifier with the bias learned through w, 
regularization parameter lambda and using 'minimizer' as an optimization function";
trainZeroOneErrorWbias::usage = "{model, margin}=trainZeroOneErrorWbias[feats, labels, lambda] 
trains a 0/1 loss classifier with regularization parameter lambda";

trainHardMarginSVM::usage = "{model, margin}=trainHardMarginSVM[Ktr, labels] trains a max-margin SVM in the dual";
train1NormSoftMarginSVM::usage = "{model, margin}=train1NormSoftMarginSVM[Ktr, labels, c] trains a 1-norm soft-margin SVM 
in the dual, using the regularization parameter c";
train2NormSoftMarginSVM::usage = "{model, margin}=train2NormSoftMarginSVM[Ktr, labels, c] trains a 2-norm soft-margin SVM 
in the dual, using the regularization parameter c";

testMaxMargin::usage = "{acc, pred, score}=testMaxMargin[model, feats, labels]] tests a linear max-margin classifier.";
testLinearClassifier::usage = "{acc, pred, score}=testLinearClassifier[w, feats, labels] tests a linear classifier with
no bias, e.g.: trainSoftMarginHingeWbias and trainZeroOneErrorWbias";
testSVM::usage = "{acc, pred, score}=testLinearClassifier[a, b, Kte, labels]";

plotData::usage = "plotData[feats, labels] is a simple module to plot a given binary dataset.";
plotLinearResults::usage = "plotLinearResults[testFunc, model, feats, labels, trErr, teErr, margin] plots the classification results";
plotKernelResults::usage = "plotKernelResults[model, fTr, yTr, fTe, yTe, kernelFunc, trErr, teErr, margin] plots the classification results";

runNoBiasExperiment::usage = "runNoBiasExperiment[fTr, yTr, fTe, yTe, classifier] runs a linear classifier experiment,
in which the specified classifier (e.g. trainSoftMarginHingeWbias, trainZeroOneErrorWbias) does not explicitly include a bias";
runMaxMarginExperiment::usage = "runMaxMarginExperiment[fTr, yTr, fTe, yTe, classifier] runs a max-margin experiment
using the specified classifier (e.g. trainMaxMargin, trainSoftMargin, trainSoftMarginHinge, or trainZeroOneError)";
runSVMExperiment::usage = "runSVMExperiment[fTr, yTr, fTe, yTe, classifier, kernelFunc] runs a Support Vector Machine experiment,
using the specified SVM classifier(eg. trainHardMarginSVM, train1NormSoftMarginSVM, train2NormSoftMarginSV), 
trained in the dual space and the specified kernel function (e.g. linearKernel, gaussianKernel).";

Begin["Private`"];

plotSize = 400;
xPos = {};
xNeg = {};

Off[Transpose::nmtx];
Off[ListPlot::argx];
Off[Show::gcomb];

loadData[filename_] :=
	Module[ {dtst, feats, labels, training, testing, fTe, yTe, fTr, yTr},
		dtst = Import[filename];
		feats = dtst[[1]][[1]];
		labels = dtst[[1]][[2]];
		testing = dtst[[1]][[3]][[1]];
		training = dtst[[1]][[4]][[1]];
		fTr = feats[[training]];
		fTe = feats[[testing]];
		yTr = labels[[training]];
		yTe = labels[[testing]];
		{fTr, yTr, fTe, yTe}
	];

loadData[x___ /; (Length[{x}] == 0 || Length[{x}] > 1)] :=
	Message[loadData::badarg, {x}]

loadData::badarg = "Only one arg please : `1`";

createData[] :=
	Module[ {plot, p, clr, s, pt},
		xPos = {};
		xNeg = {};
		plot = Plot[{}, {x, 0, 1}];
		p = {};
		clr = {};
		s = 1;
		EventHandler[
		Show[plot, Epilog -> {{PointSize[Large], Red, Point[Dynamic[xPos]]}, {PointSize[Large], Blue, Point[Dynamic[xNeg]]}}],
			"MouseDragged" :> (
				pt = MousePosition["Graphics"];
				p = Union @ Flatten[{Partition[Flatten[p], 2], Union @ Partition[pt, 2]}, 1];
				If[ s > 0,
					xPos = Union @ Flatten[{xPos, p}, 1],
					xNeg = Union @ Flatten[{xNeg, p}, 1]
				];
			),
			"MouseClicked" :> (
				p = {};
				s = s*-1
			)
		]
	];

getTrTeData[trPerc_] :=
	Module[ {lab, feat, yTr, fTr, yTe, fTe, trSet, nSamp},
		lab = Flatten[{Table[{1.}, {Length[xPos]}], Table[{-1.}, {Length[xNeg]}]}, 1];
		feat = Flatten[{xPos, xNeg}, 1];
		nSamp = Length[lab];
		SeedRandom[12345];
		trSet = Table[Boole[RandomReal[] > (100 - trPerc)/100], {i, 1, nSamp}];
		fTr = feat[[Flatten @ Position[trSet, 1], All]];
		yTr = lab[[Flatten @ Position[trSet, 1]]];
		fTe = feat[[Flatten @ Position[trSet, 0], All]];
		yTe = lab[[Flatten @ Position[trSet, 0]]];
		{fTr, yTr, fTe, yTe}
	];

hinge[margin_] :=
	Piecewise[{{1 - margin, (1 - margin) > 0}}, 0];
(*hinge[margin_] := Piecewise[{{1 - margin, margin < 1}}, 0];*)

err[margin_] :=
	Piecewise[{{1, margin < 0}}, 0];

linearKernel[fTr_, fTe_] :=
	Module[ {K},
		K = fTr.Transpose[fTe]
	];

gaussianKernel[input_] :=
	Module[ {sigmaSQ, myGaussianKern},
		sigmaSQ = If[ Length[input] > 1, estimateSigmaSQ[input], input];
		myGaussianKern[x_, y_] := computeGaussianKernel[x, y, sigmaSQ];
		myGaussianKern
	];

computeGaussianKernel[fTr_, fTe_, sigmaSQ_] :=
	Module[ {D, K},
		D = computeDist[fTr, fTe];
		K = Exp[-1/(2 sigmaSQ) D]
	];

estimateSigmaSQ[fTr_] :=
	Module[ {sigmaSQ, D},
		D = computeDist[fTr, fTr];
		sigmaSQ = Mean[Mean[D]]
	];

computeDist[fTr_, fTe_] :=
	Module[ {d, nTr, nTe, NTr, NTe, P, D},
		{nTr, d} = Dimensions[fTr];
		{nTe, d} = Dimensions[fTe];
		P = linearKernel[fTr, fTe];
		NTr = Transpose[Table[Norm /@ fTr, {i, nTe}]];
		NTe = Table[Norm /@ fTe, {i, nTr}];
		D = NTr + NTe - 2P
	];

trainMaxMargin[fTr_, yTr_] :=
	Module[ {results, model, margin, nTr, d, w, v, b, i, sol, cnstr},
		{nTr, d} = Dimensions[fTr];
		w = Table[Subscript[v, i], {i, d}];
		cnstr = And @@ (# <= 0& /@ Flatten @ (1 - (fTr.w + b) yTr));
		sol = FindMinimum[{w.w, cnstr}, Join[w, {b}], Compiled -> True, Method -> "QuadraticProgramming"] // Quiet;
		model = ({w, b} /. sol[[2]]);
		margin = 1/Sqrt[sol[[1]]];
		results = {model, margin}
	];

trainSoftMargin[fTr_List, yTr_List, regC_] :=
	Module[ {results, model, margin, nTr, d, w, v, b, xi, x, i, sol, obj, cnstr},
		{nTr, d} = Dimensions[fTr];
		w = Table[Subscript[v, i], {i, d}];
		xi = Table[Subscript[x, i], {i, nTr}];
		cnstr = And @@ (# <= 0& /@ Flatten @ (1 - xi - (fTr.w + b) yTr)) && And @@ (# >= 0& /@ Flatten @ xi);
		obj = w.w + regC Total[xi];
		sol = FindMinimum[{obj, cnstr}, Join[w, {b}, xi], Compiled -> True, Method -> "QuadraticProgramming"] // Quiet;
		model = ({w, b, xi} /. sol[[2]]);
		(*margin=1/Sqrt @ (sol[[1]] - c Total[model[[3]]])*)
		margin = (1 - Max[model[[3]]])/Norm[model[[1]]];
		results = {model, margin}
	];

trainSoftMarginHinge[feats_List, labels_List, regC_] :=
	Module[ {results, model, margin, b, d, nTr, v, w, regularizer, loss, obj, sol},
		{nTr, d} = Dimensions[feats];
		w = Table[Subscript[v, i], {i, 1, d}];
		regularizer = w.w;
		loss = Total[hinge @@@ (labels (feats.w + b))];
		obj = regularizer + regC loss;
		sol = FindMinimum[obj, Join[w, {b}]] // Quiet;
		model = ({w, b} /. sol[[2]]);
		margin = (Min[(labels (feats.model[[1]] + model[[2]]))])/Norm[model[[1]]];
		(*Print[1/(sol[[1]] - Total[hinge @@@ (1 - labels (feats.(model[[1]]) + model[[2]]))])] // N;*)
		results = {model, margin}
	];

trainZeroOneError[feats_List, labels_List, regC_] :=
	Module[ {results, model, margin, b, d, nTr, v, w, regularizer, loss, obj, sol},
		{nTr, d} = Dimensions[feats];
		w = Table[Subscript[v, i], {i, 1, d}];
		regularizer = w.w;
		loss = Total[err @@@ (labels (feats.w + b))];
		obj = regularizer + regC loss;
		sol = NMinimize[obj, Join[w,{b}], Method -> "RandomSearch"] // Quiet;
		model = ({w, b} /. sol[[2]]);
		margin = (Min[(labels (feats.model[[1]] + model[[2]]))])/Norm[model[[1]]];
		results = {model, margin}
	];

trainSoftMarginHingeWbias[feats_List, labels_List, lambda_,function_] :=
	Module[ {model, margin, results, biasFeats, d, nTr, w, v, regularizer, loss, obj, sol},
		biasFeats = Append[#,1]& /@ feats;
		{nTr, d} = Dimensions[biasFeats];
		w = Table[Subscript[v, i], {i, 1, d}];
		regularizer[u_] := lambda/2 Norm[u]^2;
		loss[x_, y_, u_] := Total[hinge @@@ (y x.u)]/nTr;
		obj[x_, y_, u_] := regularizer[u] + loss[x, y, u];
		sol = function[obj[biasFeats, labels, w], w] // Quiet;
		model = (w /. sol[[2]]);
		margin = (Min[(labels (feats.model))])/Norm[model];
		results = {model, margin}
	];

trainZeroOneErrorWbias[feats_List, labels_List, lambda_] :=
	Module[ {model, margin, results, biasFeats, d, nTr, w, v, regularizer, loss, obj, sol},
		biasFeats = Append[#, 1]& /@ feats;
		{nTr, d} = Dimensions[biasFeats];
		w = Table[Subscript[v, i], {i, 1, d}];
		regularizer[u_] := lambda/2 Norm[u]^2;
		loss[x_, y_, u_] := Total[err @@@ (y x.u)]/nTr;
		obj[x_, y_, u_] := regularizer[u] + loss[x, y, u];
		sol = NMinimize[obj[biasFeats, labels, w], w] // Quiet;
		model = (w /. sol[[2]]);
		margin = (Min[(labels (feats.model))])/Norm[model];
		results = {model, margin}
	];

trainHardMarginSVM[KTr_, yTr_] :=
	Module[ {nTr, d, H, f, a, alpha, b, margin, sol, obj, constraints},
		{nTr, d} = Dimensions[KTr];
		f = Table[1, {i, nTr}];
		alpha = Table[Subscript[a, i], {i, nTr}];
		H = yTr.Transpose[yTr] KTr;
		constraints = First[alpha.yTr] == 0 && (# >= 0& /@ (And @@ alpha));
		obj = 1/2 alpha.H.alpha - f.alpha;
		sol = FindMinimum[{obj, constraints}, alpha, Compiled -> True, AccuracyGoal -> 1, PrecisionGoal -> 1, MaxIterations -> 100, Method -> "QuadraticProgramming", Gradient :> H.alpha - f];
		alpha = (alpha /. sol[[2]]);
		alpha[[Flatten @ Position[# < 10^(-8)& /@ alpha, True]]] = 0;
		b = -1/Total[alpha] (alpha yTr[[All,1]]).H.alpha;
		margin = Total[alpha]^(-1/2);
		(*margin=(-sol[[1]])^(-1/2);*)
		alpha = alpha yTr[[All,1]];
		{{alpha, b}, margin}
	];

train1NormSoftMarginSVM[KTr_, yTr_, regC_] :=
	Module[ {nTr, d, H, f, a, alpha, b, nrm, margin, sol, obj, constraints},
		{nTr, d} = Dimensions[KTr];
		f = Table[1, {i, nTr}];
		alpha = Table[Subscript[a, i], {i, nTr}];
		H = yTr.Transpose[yTr] KTr;
		constraints = First[alpha.yTr] == 0 && (# >= 0& /@ (And @@ alpha)) && (# <= regC& /@ (And @@ alpha));
		obj = 1/2 alpha.H.alpha - f.alpha;
		sol = FindMinimum[{obj, constraints}, alpha, Compiled -> True, AccuracyGoal -> 1, PrecisionGoal -> 1, MaxIterations -> 100, Method -> "QuadraticProgramming", Gradient :> H.alpha - f];
		alpha = (alpha /. sol[[2]]);
		alpha[[Flatten @ Position[# < 10^(-8)& /@ alpha, True]]] = 0;
		b = -1/Total[alpha] (alpha yTr[[All,1]]).H.alpha;
		nrm = (2(sol[[1]] + Total[alpha]))^(1/2);
		alpha = alpha yTr[[All,1]];
		margin = (Min[(yTr(KTr.alpha + b))])/nrm;
		{{alpha, b}, margin}
	];

train2NormSoftMarginSVM[KTr_, yTr_, regC_] :=
	Module[ {model, nrm, margin, nTr},
		{nTr, nTr} = Dimensions[KTr];
		{model, margin} = trainHardMarginSVM[KTr + 1/regC IdentityMatrix[nTr], yTr];
		nrm = (margin^(-2) - 1/regC Norm[model[[1]]]^2)^(1/2);
		margin = (1 - Max[yTr model[[1]]]/regC)/nrm;
		{model, margin}
	];

testMaxMargin[model_, feats_, labels_] :=
	Module[ {w, b, nTe, d, pred, acc, score},
		{nTe, d} = Dimensions[feats];
		w = model[[1]];
		b = model[[2]];
		score = feats.w + b;
		pred = Sign[score] // N;
		acc = Count[# > 0& /@ (pred labels[[All, 1]]), True]/Length[labels] // N;
		{acc, pred, score}
	];

testLinearClassifier[w_, feats_, labels_] :=
	Module[ {biasFeats, nTe, d, pred, acc, score},
		biasFeats = Append[#, 1]& /@ feats;
		{nTe, d} = Dimensions[biasFeats];
		score = biasFeats.w;
		pred = Sign[score] // N;
		acc = Count[# > 0& /@ (pred labels[[All, 1]]), True]/Length[labels] // N;
		{acc, pred, score}
	];

testSVM[model_, KTe_, yTe_] :=
	Module[ {nTe, nTr, a, b, pred, score, correct},
		a = model[[1]];
		b = model[[2]];
		{nTe, nTr} = Dimensions[KTe];
		score = Flatten[KTe.a + b];
		pred = Sign[score];
		correct = Count[# > 0& /@ (score yTe[[All, 1]]), True]/Length[yTe] // N;
		{correct, pred, score}
	];

plotData[feats_List, labels_List] :=
	Module[ {pos, neg},
		pos = Transpose[MatrixForm[Position[labels, 1.]]][[All, 1]][[1]];
		neg = Transpose[MatrixForm[Position[labels, -1.]]][[All, 1]][[1]];
		ListPlot[{feats[[neg]], feats[[pos]]}, PlotRange -> Full, ImageSize -> plotSize, PlotRange -> {{Min[feats], Max[feats]}, {Min[feats], Max[feats]}}]
	];

plotLinearResults[testFunc_, model_, feats_List, labels_List, trErr_Real, teErr_Real, margin_Real] :=
	Module[ {pos, neg, a, b},
		pos = Transpose[MatrixForm[Position[labels, 1.]]][[All, 1]][[1]];
		neg = Transpose[MatrixForm[Position[labels, -1.]]][[All, 1]][[1]];
		a = ListPlot[{feats[[neg]], feats[[pos]]}, PlotRange -> Full];
		b = ContourPlot[testFunc[model, {{x, y}}, {{1}}][[3]], {x, Min[feats[[All, 1]]], Max[feats[[All, 1]]]}, {y, Min[feats[[All, 2]]], Max[feats[[All, 2]]]}, Contours -> {0}, PlotPoints -> 4];
		Show[b, a, ImageSize -> plotSize, PlotLabel -> Style[ StringJoin[{"TRerr=", ToString[NumberForm[trErr, 3]], "% TEerr=", ToString[NumberForm[teErr, 3]], "% Marg=", ToString[NumberForm[margin, 3]]}], FontSize -> 21]]
	];

plotLinearResults[testFunc_, model_, fTr_List, yTr_List, fTe_List, yTe_List, trErr_Real, teErr_Real, margin_Real] :=
	Module[ {f1, f2, posTr, negTr, posTe, negTe, a, b, c},
		posTr = Transpose[MatrixForm[Position[yTr, 1.]]][[All, 1]][[1]];
		negTr = Transpose[MatrixForm[Position[yTr, -1.]]][[All, 1]][[1]];
		posTe = Transpose[MatrixForm[Position[yTe, 1.]]][[All, 1]][[1]];
		negTe = Transpose[MatrixForm[Position[yTe, -1.]]][[All, 1]][[1]];
		a = ListPlot[{fTr[[negTr]], fTr[[posTr]]}, PlotRange -> Full, PlotMarkers -> {Automatic, 24}];
		c = ListPlot[{fTe[[negTe]], fTe[[posTe]]}, PlotRange -> Full];
		f1 = Join[fTr[[All, 1]], fTe[[All, 1]]];
		f2 = Join[fTr[[All, 2]], fTe[[All, 2]]];
		b = ContourPlot[testFunc[model, {{x, y}}, {{1}}][[3]], {x, Min[f1], Max[f1]}, {y, Min[f2], Max[f2]}, Contours -> {0}, PlotPoints -> 4];
		Show[b, a, c, ImageSize -> plotSize, PlotLabel -> Style[ StringJoin[{"TRerr=", ToString[NumberForm[trErr,3]], "% TEerr=", ToString[NumberForm[teErr,3]], "% Marg=", ToString[NumberForm[margin, 3]]}], FontSize -> 21]]
	];

plotKernelResults[model_, fTr_, yTr_, fTe_, yTe_, kernelFunc_, trErr_, teErr_, margin_] :=
	Module[ {sv, pos, posSV, negSV, posAll, negAll, fnSV, fpSV, a, b, d},
		sv = Boole[# > 0]& /@ (model[[1]] Flatten @ yTr);
		pos = Boole[# > 0]& /@ Flatten @ yTr;
		posSV = Transpose[MatrixForm[Position[(sv pos), 1]]][[All, 1]][[1]];
		negSV = Transpose[MatrixForm[Position[(sv (1 - pos)), 1]]][[All, 1]][[1]];
		posAll = Transpose[MatrixForm[Position[pos, 1]]][[All, 1]][[1]];
		negAll = Transpose[MatrixForm[Position[(1 - pos), 1]]][[All, 1]][[1]];
		(*posTe=Transpose[MatrixForm[Position[yTe, 1.]]][[All, 1]][[1]];
		negTe=Transpose[MatrixForm[Position[yTe, -1.]]][[All, 1]][[1]];*)
		fnSV = fTr[[negSV]];
		fpSV = fTr[[posSV]];
		fnSV = If[ Length[fnSV] == 0, {{Null, Null}}, fnSV];
		fpSV = If[ Length[fpSV] == 0, {{Null, Null}}, fpSV];
		a = ListPlot[{fnSV, fpSV}, PlotRange -> Full, PlotMarkers -> {Automatic, 24}];
		b = ListPlot[{fTr[[negAll]], fTr[[posAll]]}, PlotRange -> Full];
		(*b=Labeled[ListPlot[{fTr[[negAll]], fTr[[posAll]]}, PlotRange -> Full], StringJoin["Classification results. #SV", ToString[Length[model[[1]]]]]];*)
		(*c=ListPlot[{fTe[[negTe]], fTe[[posTe]]}, PlotRange -> Full];*)
		d = ContourPlot[First[(kernelFunc[{{x, y}}, fTr]).model[[1]] + model[[2]]], {x, Min[fTr[[All, 1]]], Max[fTr[[All, 1]]]},
		{y, Min[fTr[[All, 2]]], Max[fTr[[All, 2]]]}, Contours -> {0}, PlotPoints -> 50];
		Show[d, a, b, ImageSize -> plotSize, PlotLabel -> Style[ StringJoin[{"TRerr=", ToString[NumberForm[trErr, 3]], "% TEerr=", ToString[NumberForm[teErr, 3]], "% Marg=", ToString[NumberForm[margin, 3]]}], FontSize -> 21]]
	];

runMaxMarginExperiment[fTr_List, yTr_List, fTe_List, yTe_List, classifier_] :=
	Module[ {model, margin, teResults, trResults},
		{model, margin} = classifier[fTr, yTr];
		trResults = testMaxMargin[model, fTr, yTr];
		teResults = testMaxMargin[model, fTe, yTe];
		plotLinearResults[testMaxMargin, model, fTr, yTr, fTe, yTe, (1 - trResults[[1]])*100, (1 - teResults[[1]])*100, margin]
		(*plotLinearResults[testMaxMargin, model, Join[fTr, fTe], Join[yTr, yTe], (1 - trResults[[1]])*100, (1 - teResults[[1]])*100, margin]*)
		(*teResults[[1]]*100*)
	];

runNoBiasExperiment[fTr_List, yTr_List, fTe_List, yTe_List, classifier_] :=
	Module[ {model, margin, trResults, teResults},
		(*{model, margin} = trainSoftMarginHingeWbias [1/c, fTr, yTr, FindMinimum];*)
		(*{model, margin} = trainZeroOneErrorWbias[lambda, fTr, yTr];*)
		{model, margin} = classifier[fTr, yTr];
		trResults = testLinearClassifier[model, fTr, yTr];
		teResults = testLinearClassifier[model, fTe, yTe];
		plotLinearResults[testLinearClassifier, model, fTr, yTr, fTe, yTe, (1 - trResults[[1]])*100, (1 - teResults[[1]])*100, margin]
		(*plotLinearResults[testLinearClassifier, model, Join[fTr, fTe],Join[yTr, yTe],(1 - trResults[[1]])*100,(1 - teResults[[1]])*100,margin]*)
		(*teResults[[1]]*100*)
	];

runSVMExperiment[fTr_List, yTr_List, fTe_List, yTe_List, classifier_, kernFunc_] :=
	Module[ {KTr, model, margin, trResults, teResults},
		KTr = kernFunc[fTr, fTr];
		{model, margin} = classifier[KTr, yTr];
		trResults = testSVM[model, KTr, yTr];
		teResults = testSVM[model, kernFunc[fTe, fTr], yTe];
		plotKernelResults[model, fTr, yTr, fTe, yTe, kernFunc, (1 - trResults[[1]])*100, (1 - teResults[[1]])*100, margin]
	];

End[];

Protect[
	plotSize, loadData, createData, getTrTeData, hinge, err,
	linearKernel, gaussianKernel, computeGaussianKernel, estimateSigmaSQ, computeDist,
	trainMaxMargin, trainSoftMargin, trainSoftMarginHinge, trainZeroOneError,
	trainSoftMarginHingeWbias, trainZeroOneErrorWbias,
	trainHardMarginSVM, train1NormSoftMarginSVM, train2NormSoftMarginSVM,
	testMaxMargin, testLinearClassifier, testNonLinearSVM,
	plotData, plotLinearResults, plotKernelResults,
	runNoBiasExperiment, runMaxMarginExperiment, runSVMExperiment
];

EndPackage[];

Print["SVM Package Loaded"];
