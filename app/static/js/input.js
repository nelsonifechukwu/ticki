const imgInputHelper = document.getElementById("input-img");
const imgInputHelperLabel = document.getElementById("input-img-label");
const imgContainer = document.querySelector(".input-img-container");
const imgFiles = [];

const addImgHandler = () => {
    const file = imgInputHelper.files[0];
    if (!file) return;
    // Generate img preview
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      const newImg = document.createElement("img");
      newImg.src = reader.result;
      imgContainer.insertBefore(newImg, imgInputHelperLabel);
    };
    // Store img file
    imgFiles.push(file);
    // Reset image input
    imgInputHelper.value = "";
    return;
  };
  
  imgInputHelper.addEventListener("change", addImgHandler);