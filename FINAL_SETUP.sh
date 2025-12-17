#!/bin/bash

echo "🧠 Brain Tumor Dashboard - Final Setup"
echo "======================================"

# Create backend .env
echo "Creating backend .env..."
cat > backend/.env << 'BACKEND_ENV'
PORT=5000
MONGODB_URI=mongodb://localhost:27017/brain-tumor
NODE_ENV=development
BACKEND_ENV

echo "✓ Backend .env created"

# Create frontend .env
echo "Creating frontend .env..."
cat > frontend/.env << 'FRONTEND_ENV'
VITE_API_URL=http://localhost:5000/api
FRONTEND_ENV

echo "✓ Frontend .env created"

echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo ""
echo "Next Steps:"
echo ""
echo "1. Start MongoDB (Terminal 1):"
echo "   brew services start mongodb-community  # macOS"
echo "   sudo systemctl start mongod            # Linux"
echo ""
echo "2. Start Backend (Terminal 2):"
echo "   cd backend && npm start"
echo ""
echo "3. Start Frontend (Terminal 3):"
echo "   cd frontend && npm run dev"
echo ""
echo "4. Open Dashboard:"
echo "   http://localhost:5173"
echo ""
echo "======================================"
