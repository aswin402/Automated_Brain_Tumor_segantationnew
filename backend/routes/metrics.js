import express from 'express'
import { getMetrics, initializeMetrics, getMetricByModel } from '../controllers/metricController.js'

const router = express.Router()

router.get('/', getMetrics)
router.post('/initialize', initializeMetrics)
router.get('/:model', getMetricByModel)

export default router
