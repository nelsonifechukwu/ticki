const imgInput = document.getElementById("input-img-preview");
const imgInputLabel = document.getElementById("input-img-label");
const imgContainer = document.querySelector(".input-img-container");
let imgFiles = [];

//----Implement search for multiple images later on----------

// const addImgHandler = () => {
//     const file = imgInputHelper.files[0];
//     if (!file) return;
//     // Generate img preview
//     const reader = new FileReader();
//     reader.readAsDataURL(file);
//     reader.onload = () => {
//       const newImg = document.createElement("img");
//       newImg.src = reader.result;
//       if imgContainer.>1: return;
//       imgContainer.insertBefore(newImg, imgInputHelperLabel);
//     };

//     // Store img file
//     imgFiles.push(file);
//     // Reset image input
//     imgInputHelper.value = "";
//     return;
//   };
const previewImgHandler = () => {
  //we use imgFiles to hold the actual img, since it cannot be
  //reconstructed from newImg.src
  const file = imgInput.files[0];
  if (!file) return;

  // Generate img preview
  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onload = () => {
    const newImg = document.createElement("img");
    newImg.src = reader.result; //get the input img

    // Check if imgContainer already contains an img element
    const existingImg = imgContainer.querySelector("img");
    if (existingImg) {
      // If there's already an img element, replace it with the new input
      imgContainer.replaceChild(newImg, existingImg);
      imgFiles[0] = file;
    } else {
      // If no img element, insert the new input before the label
      imgContainer.insertBefore(newImg, imgInputLabel);
      imgFiles = [file];
    }
  };
  // Reset image input
  //imgInput.value = "";
};

const customFormSubmitHandler = async (ev) => {
  ev.preventDefault(); //prevent form submission
  const _ImgInput = document.getElementById("q-img");

  _ImgInput.files = imgInput.files;
  form = ev.target; // don't submit form to server w/.submit()

  // Prepare form data
  const formData = new FormData(form);
  try {
    const response = await fetch(form.action, {
      method: "POST",
      body: formData,
    });

    const html = await response.text();
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, "text/html");

    // Find the newly rendered selected-imgs div from response
    const newImgsDiv = doc.querySelector(".selected-imgs");
    const currentImgsDiv = document.querySelector(".selected-imgs");

    if (newImgsDiv && currentImgsDiv) {
      currentImgsDiv.innerHTML = newImgsDiv.innerHTML;
    } else {
      console.warn("Could not find .selected-imgs in response.");
    }
  } catch (err) {
    console.error("Failed to submit form:", err);
  }
};

imgInput.addEventListener("change", previewImgHandler);
document
  .querySelector(".img-input-form")
  .addEventListener("submit", customFormSubmitHandler);
