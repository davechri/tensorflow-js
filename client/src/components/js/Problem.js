import { assert } from '@tensorflow/tfjs-core/dist/util';
import Global from './Global';
import {LossFunction, ProblemType} from './Constants';

class HousePriceProblem {
  modelType = 'Regression';
  storageKey = 'localstorage://house_price_model';
  classNames = [];
  csvColumn = undefined;
  label = 'Price';

  filterPoint(point) {return true;}
}

class WaterFrontProblem {
  modelType = 'Binary Classification';
  storageKey = 'localstorage://water_front_model';
  classNames = ['0','1'];
  csvColumn = 'waterfront';
  label = 'Water front';

  getClassName(value) {return this.classNames[value];}

  filterPoint(point) {return true;}
}

class BedroomsProblem {
  modelType = 'Multiclass Classification';
  storageKey = 'localstorage://bedrooms_model';
  classNames = ['1', '2', '3+'];
  csvColumn = 'bedrooms';
  label='Bedrooms';

  getClassName(value) {
    value = Math.min(value, this.classNames.length);
    return this.classNames[value-1];
  }

  filterPoint(point) {return point.bedrooms > 0;}
}

class Problem {
  constructor() {
    this.problem = new HousePriceProblem();
  }

  change(problemType) {
    switch(problemType) {
      case ProblemType.HOUSE_PRICE:
        this.problem = new HousePriceProblem();
        Global.setLossFunction(LossFunction.MEAN_SQUARED_ERROR);
        Global.setLearningRate(0.5);
        break;
      case ProblemType.WATER_FRONT:
        this.problem = new WaterFrontProblem();
        Global.setLossFunction(LossFunction.BINARY_CROSS_ENTROPY);
        Global.setLearningRate(0.01);
        break;
      case ProblemType.BEDROOMS:
        this.problem = new BedroomsProblem();
        Global.setLossFunction(LossFunction.CATEGORICAL_CROSS_ENTROPY);
        Global.setLearningRate(0.01);
        break;
      default:
        assert('unknown problem type', problemType);
        break;
    }
  }

  getModelType() {
    return this.problem.modelType;
  }

  filterPoint(point) {
    return this.problem.filterPoint(point);
  }

  isClassification() {
    return this.problem.classNames.length > 0;
  }

  isBinaryClassification() {
    return this.problem.classNames.length === 2;
  }

  isMultiClassClassification() {
    return this.problem.classNames.length > 2;
  }

  getStorageKey() {
    return this.problem.storageKey;
  }

  getClassNames() {
    return this.problem.classNames;
  }

  getClassName(point) {
    const value = point[this.problem.csvColumn];
    return this.problem.getClassName(value);
  }

  getCsvColumn() {
    return this.problem.csvColumn;
  }

  getLabel() {
    return this.problem.label;
  }
}

export default Problem = new Problem();