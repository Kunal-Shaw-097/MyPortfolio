
function display(){
    var imgcanvas = document.getElementById("canv1");
    imgcanvas.height = 540
    imgcanvas.width = 540
    var fileinput = document.getElementById("finput");
    var img = document.getElementById('temp');
    if (img){
        img.src = '';
        img.style.display = 'none';
    } 
    var image = new SimpleImage(fileinput);
    image.drawTo(imgcanvas);
}