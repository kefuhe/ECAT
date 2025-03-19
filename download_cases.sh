```bash
#!/bin/bash

# Define the target directory for the cases repository
CASES_DIR="ECAT-cases"

# Check if the directory already exists
if [ -d "$CASES_DIR" ]; then
    echo "The ECAT-cases directory already exists. Pulling the latest changes..."
    cd "$CASES_DIR"
    git pull
    cd ..
else
    echo "Cloning the ECAT-cases repository..."
    git clone https://github.com/kefuhe/ECAT-cases.git "$CASES_DIR"
fi

echo "ECAT-cases repository is ready in the '$CASES_DIR' directory."