import mongoose from 'mongoose'

const modelSchema = new mongoose.Schema(
  {
    name: String,
    displayName: String,
    type: String,
    accuracy: Number,
    precision: Number,
    recall: Number,
    f1: Number,
    trainedAt: Date,
    status: {
      type: String,
      default: 'active'
    }
  },
  { timestamps: true }
)

const Model = mongoose.model('Model', modelSchema)

export default Model
