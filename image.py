from PIL import Image

# Load the images
image1 = Image.open('RawData/bestImageSim.png')
image2 = Image.open('RawData/real-fixed.jpg')

# Get the dimensions of the images
width1, height1 = image1.size
width2, height2 = image2.size

# Determine the new width and height for the images
new_width = min(width1, width2)
new_height1 = int((new_width / width1) * height1)
new_height2 = int((new_width / width2) * height2)

# Resize the images
image1 = image1.resize((new_width, new_height1), Image.ANTIALIAS)
image2 = image2.resize((new_width, new_height2), Image.ANTIALIAS)

# Create a new image with the combined height and the new width
new_image = Image.new('RGB', (new_width, new_height1 + new_height2))

# Paste the images into the new image
new_image.paste(image1, (0, 0))
new_image.paste(image2, (0, new_height1))

# Save the new image
new_image.save('Produced_Plots/combined_image.png')

# Load the images
image3 = Image.open('Produced_Plots/FFTSIM/HOPG_Lattice_Reciprocal_Space_3nm.png')
image4 = Image.open('RawData/k-vec.jpg')

# Get the dimensions of the images
width3, height3 = image3.size
width4, height4 = image4.size

# Determine the new width and height for the images
new_width2 = min(width3, width4)
new_height3 = int((new_width2 / width3) * height3)
new_height4 = int((new_width2 / width4) * height4)

# Resize the images
image3 = image3.resize((new_width2, new_height3), Image.ANTIALIAS)
image4 = image4.resize((new_width2, new_height4), Image.ANTIALIAS)

# Create a new image with the combined height and the new width
new_image2 = Image.new('RGB', (new_width2, new_height3 + new_height4))

# Paste the images into the new image
new_image2.paste(image3, (0, 0))
new_image2.paste(image4, (0, new_height3))

# Save the new image
new_image2.save('Produced_Plots/combined_image2.png')