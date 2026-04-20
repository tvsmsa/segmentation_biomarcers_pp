from PIL import Image

image_id = 6451
file_name = f"{image_id}_3D-Macula-6x6_RETINA01_Landscape_001"
index = image_id
img = Image.open(f"dataset/{index}/{file_name}.jpg")
res = img.crop((37, 220, 450, 630))
#res.show()
res.save(f"dataset/{index}/{file_name}_1.jpg")
res = img.crop((37, 672, 450, 1082))
#res.show()
res.save(f"dataset/{index}/{file_name}_2.jpg")
res = img.crop((496, 222, 1032, 630))
#res.show()
res.save(f"dataset/{index}/{file_name}_3.jpg")
res = img.crop((496, 672, 1032, 1082))
#res.show()
res.save(f"dataset/{index}/{file_name}_4.jpg")
res = img.crop((1340, 260, 1582, 505))
#res.show()
res.save(f"dataset/{index}/{file_name}_5.jpg")