printf '%s\n' ""

printf '%s\n' "Creating Python virtual environment..."
python -m venv venv && printf '%s\n' "Virtual environment was successfully created." || printf '%s\n' "An error occurred while creating the virtual environment."

printf '%s\n' "Activating virtual environment..."
if . ./venv/bin/activate; then
    printf '%s\n' "Virtual environment was successfully initialized."
else
    printf '%s\n' "An error occurred while initializing the virtual environment."
fi

printf '%s\n' "Installing required packages..."
if python -m pip install -r ./requirements.txt; then
    printf '%s\n' "Required packages were successfully installed."
else
    printf '%s\n' "An error occurred while installing the required packages."
fi
