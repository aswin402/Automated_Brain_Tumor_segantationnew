import express from 'express'
import { getModels, initializeModels, getModelByName } from '../controllers/modelController.js'

const router = express.Router()

router.get('/', getModels)
router.post('/initialize', initializeModels)
router.get('/:name', getModelByName)

export default router
