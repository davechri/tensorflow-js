export class LossFunction {
  static BINARY_CROSS_ENTROPY = 'binaryCrossentropy';
  static CATEGORICAL_CROSS_ENTROPY = 'categoricalCrossentropy';
  static MEAN_SQUARED_ERROR = 'meanSquaredError';
}

export default class Constants {
  static LOSS_NOT_SET = 0x7fffffff;
}

export class ProblemType {
  static HOUSE_PRICE = 'House price'; // regression
  static WATER_FRONT = 'Water front'; // binary classification
  static BEDROOMS = 'Bedrooms'; // multi class
}