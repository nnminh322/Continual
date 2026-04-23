#!/bin/bash
# Universal Linux Script - Adds CDP Port to Antigravity

echo "=== Antigravity CDP Setup ==="
echo ""
echo "Searching for Antigravity shortcuts..."

IDE_NAME="Antigravity"
IDE_NAME_LOWER=$(echo "$IDE_NAME" | tr '[:upper:]' '[:lower:]')

# Define search locations for .desktop files
SEARCH_LOCATIONS=(
    "$HOME/.local/share/applications"
    "$HOME/Applications"
    "$HOME/.config/autostart"
    "/usr/share/applications"
    "/usr/local/share/applications"
    "/var/lib/snapd/desktop/applications"
    "/var/lib/flatpak/exports/share/applications"
)

# Function to add CDP port to a .desktop file
add_cdp_to_desktop_file() {
    local desktop_file="$1"
    local backup_file="${desktop_file}.bak"

    # Check if CDP port already exists
    if grep -q "remote-debugging-port" "$desktop_file"; then
        echo "  Status: CDP port already present"
        return 0
    fi

    # Create backup
    cp "$desktop_file" "$backup_file"
    echo "  Backup created: $backup_file"

    # Add CDP port to Exec lines
    sed -i 's|^Exec=\(.*\)$|Exec=\1 --remote-debugging-port=9000|' "$desktop_file"

    # Add to TryExec if present
    if grep -q "^TryExec=" "$desktop_file"; then
        sed -i 's|^TryExec=\(.*\)$|TryExec=\1 --remote-debugging-port=9000|' "$desktop_file"
    fi

    echo "  Status: CDP port added"
    return 0
}

found_count=0

# Search for .desktop files
for dir in "${SEARCH_LOCATIONS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Searching: $dir"

        for file in "$dir"/*.desktop; do
            if [ -f "$file" ]; then
                # Check if file contains the IDE name
                if grep -qi "$IDE_NAME_LOWER" "$file" 2>/dev/null; then
                    echo ""
                    echo "---"
                    echo "Found: $(basename "$file")"
                    echo "Location: $file"

                    found_count=$((found_count + 1))
                    add_cdp_to_desktop_file "$file"
                fi
            fi
        done
    fi
done

echo ""
echo "=== Setup Complete ==="
echo "Total shortcuts found: $found_count"

if [ $found_count -eq 0 ]; then
    echo ""
    echo "No shortcuts found for '$IDE_NAME'."
    echo "Please make sure Antigravity is installed."
else
    echo ""
    echo "Please restart Antigravity completely for changes to take effect."
fi