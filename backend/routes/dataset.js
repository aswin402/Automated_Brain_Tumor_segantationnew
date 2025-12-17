import express from 'express'
import { getDatasetInsights, getConfusionMatrix } from '../controllers/datasetController.js'

const router = express.Router()

router.get('/insights', getDatasetInsights)
router.get('/confusion-matrix/:model', getConfusionMatrix)

export default router
