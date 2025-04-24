#!/bin/bash

# Update repository script
echo "🚀 Updating GitHub repository..."

# Add all new files and changes
echo "📂 Adding new files and changes..."
git add .

# Commit the changes
echo "💾 Committing changes..."
git commit -m "Add data visualization for model results and Docker improvements"

# Push to GitHub
echo "🔄 Pushing to GitHub..."
git push origin main

echo "✅ Repository updated successfully!"
echo "🌐 Visit your repository at: https://github.com/Jaecy03/Startup-Success"
