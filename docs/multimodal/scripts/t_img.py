import re,sys
from io import BytesIO
from PIL import Image
src=open(str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'README.md')).read()
code=re.search(r'```python\n(from io import BytesIO.*?)```',src,re.S).group(1)
exec(code)
def mk(mode,size,fmt,**kw):
    b=BytesIO(); Image.new(mode,size,**kw).save(b,format=fmt); return b.getvalue()
# RGBA transparent -> JPEG white bg
out,mt=prepare_image(mk("RGBA",(4000,1000),"PNG",color=(0,0,0,0)))
im=Image.open(BytesIO(out)); assert mt=="image/jpeg" and im.size==(1568,392) and im.getpixel((10,10))[0]>250, im.getpixel((10,10))
# PNG keeps alpha
out,mt=prepare_image(mk("RGBA",(100,100),"PNG",color=(0,0,0,0)),fmt="PNG")
im=Image.open(BytesIO(out)); assert mt=="image/png" and im.mode=="RGBA" and im.getpixel((1,1))[3]==0
# P with transparency
p=Image.new("P",(50,50)); p.info["transparency"]=0; b=BytesIO(); p.save(b,format="PNG",transparency=0)
out,mt=prepare_image(b.getvalue()); assert Image.open(BytesIO(out)).getpixel((1,1))[0]>250
# EXIF orientation
im=Image.new("RGB",(200,100),"red"); ex=Image.Exif(); ex[0x0112]=6; b=BytesIO(); im.save(b,format="JPEG",exif=ex)
out,_=prepare_image(b.getvalue()); o=Image.open(BytesIO(out)); assert o.size==(100,200) and not o.getexif(), o.size
# CMYK, 16-bit
out,_=prepare_image(mk("CMYK",(30,30),"JPEG")); out,_=prepare_image(mk("I;16",(30,30),"PNG"),fmt="PNG")
print("ALL OK")
