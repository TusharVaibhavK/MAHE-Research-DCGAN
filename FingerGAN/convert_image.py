from PIL import Image

# Paths
input_path = "data/SOCOFing/Real/9__M_Right_thumb_finger.BMP"  # Original BMP file
output_path = "data/SOCOFing/Real/sample_fingerprint.png"      # Output PNG file

# Convert BMP to PNG
try:
    img = Image.open(input_path)
    img.save(output_path, "PNG")
    print(f"Successfully converted {input_path} to {output_path}")
except FileNotFoundError:
    print(f"Error: Could not find {input_path}. Please check the file exists.")
except Exception as e:
    print(f"Error during conversion: {e}")
