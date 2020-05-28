import {setGlobal} from 'reactn';
import GlobalSettings from './GlobalSettings';
import Constants, {ProblemType} from './Constants';

setGlobal({
  problemType: ProblemType.HOUSE_PRICE,
  dataSetSize: 0,
  modelSaved: false,
  trainingElapsedTime: 0,
  currentEpoch: 0,
  minTrainingLoss: Constants.LOSS_NOT_SET,
  currentTrainingLoss: Constants.LOSS_NOT_SET,
  trainingInprogess: false,
});

class Global extends GlobalSettings {
  constructor() {
    super();
  }

  setProblemType(p) {
    setGlobal({problemType: p});
  }
}

export default Global = new Global();