import {setGlobal, getGlobal} from 'reactn';
import Model from '../tf/Model';
import Data from '../tf/Data';
import Weights from '../Weights';
import {ProblemType} from './Constants';

export default class GlobalSettings {
  constructor() {
    setGlobal({
      problemType: ProblemType.HOUSE_PRICE,
      hiddenLayers: 10,
      activationFunction: 'sigmoid',
      lossFunction: 'meanSquaredError',
      optimizer: 'adam',
      learningRate: 0.05,
      trainingToTestingRatio: 50,
      minDeltaLoss: 0,
      batchSize: 32
    });
  }

  setHiddenLayers(h) {
    setGlobal({hiddenLayers: Number(h)});
    if(getGlobal().hiddenLayers !== h && h  > 0) {
      Model.create();
    }
  }

  setActivationFunction(a) {
    if(getGlobal().activationFunction !== a) {
      setGlobal({activationFunction: a});
      const oldModel = Model.model;
      Model.create();
      if(oldModel !== undefined) {
        Weights.copy(Model.model.weights, oldModel.model.weights);
      }
    }
  }

  setLossFunction(l) {
    if(getGlobal().lossFunction !== l) {
      setGlobal({lossFunction: l});
      Model.compile();
    }
  }

  setOptimizer(o) {
    if(getGlobal().optimizer !== o) {
      setGlobal({optimizer: o});
      Model.compile();
    }
  }

  setLearningRate(r) {
    setGlobal({learningRate: r});
    if(getGlobal().learningRate !== r && r > 0) {
      Model.compile();
    }
  }

  setTrainingToTestingRatio(r) {
    setGlobal({trainingToTestingRatio: Number(r)});
    Data.split(r);
  }

  setMinDeltaLoss(e) {
    setGlobal({minDeltaLoss: Number(e)});
  }

  setBatchSize(b) {
    setGlobal({batchSize: Number(b)});
  }
}