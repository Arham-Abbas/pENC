# List of OpenCL dependencies to search for
$OpenCLList = @(
    "OpenCL*"
)

# Convert the list of OpenCL into a regex pattern
$OpenCLPattern = $OpenCLList -join '|'

# Function to search for OpenCL matching the pattern
function Search-OpenCL {
    param (
        [string]$path
    )
    Get-ChildItem -Path $path -Recurse -Filter OpenCL* -ErrorAction SilentlyContinue -Force | Where-Object {
        $_.Name -match $OpenCLPattern
    } | ForEach-Object {
        Write-Output "Found $($_.Name) at $($_.FullName)"
    }
}

# Search for OpenCL in the C: drive
Search-OpenCL -path "C:\"
