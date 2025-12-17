import mongoose from 'mongoose'

const metricSchema = new mongoose.Schema(
  {
    name: String,
    modelName: String,
    accuracy: Number,
    precision: Number,
    recall: Number,
    f1: Number,
    auc: Number,
    supportNo: Number,
    supportGlioma: Number,
    supportMeningioma: Number,
    supportPituitary: Number
  },
  { timestamps: true }
)

const Metric = mongoose.model('Metric', metricSchema)

export default Metric
