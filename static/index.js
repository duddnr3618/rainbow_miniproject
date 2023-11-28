
function mainInit() {
    if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia )
    {
        alert("Media Device not supported")
        return;
    }
    navigator.mediaDevices.getUserMedia({video:true})
        .then(camInit)
        .catch(camInitFailed);
}