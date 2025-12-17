import mongoose from 'mongoose'

const predictionSchema = new mongoose.Schema(
  {
    filename: String,
    imageName: String,
    predictedClass: String,
    predicted_class: String,
    confidence: Number,
    probabilities: {
      no_tumor: Number,
      glioma: Number,
      meningioma: Number,
      pituitary: Number
    },
    model: {
      type: String,
      default: 'xgboost'
    },
    imageData: {
      type: String,
      default: null
    }
  },
  { timestamps: true }
)

const Prediction = mongoose.model('Prediction', predictionSchema)

export default Prediction
