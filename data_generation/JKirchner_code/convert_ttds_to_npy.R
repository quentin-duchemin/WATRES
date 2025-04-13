library(reticulate)

# Define the root directory and site names
root <- "../"
# sites <- c("Pully_small_storage", "Pully_large_storage", "Lugano_small_storage", "Lugano_large_storage", "Basel_small_storage", "Basel_large_storage")
sites <- c("Pully_small_storage")

# Loop through each site and process the .rda file
for (site in sites) {
    # Construct the file path to the .rda file
    file_path <- file.path(root, site, "data", "TTD.rda")
    
    # Load the .rda file
    load(file_path)
    
    # Ensure the TTD object exists (update with actual object name if different)
    if (exists("TTD")) {
        
        # Construct the output path
        output_file <- file.path(root, site, "data", "TTD.npy")
        TTD <- as.data.frame(TTD)

        # Use NumPy to save the tensor
        np <- import("numpy")
        np$save(output_file, TTD)

    } else {
        cat("Object 'TTD' not found in file:", file_path, "\n")
    }
}