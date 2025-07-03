const img_input = document.getElementById("input-img-preview");
const img_input_label = document.getElementById("input-img-label");
const img_container = document.querySelector(".input-img-container");
const form_element = document.querySelector(".img-input-form");

const preview_img_handler = () => {
  const file = img_input.files[0];
  if (!file) return;

  // Generate img preview
  const reader = new FileReader();
  reader.readAsDataURL(file);

  reader.onload = () => {
    const new_img = document.createElement("img");
    new_img.src = reader.result; //get the input img

    // Check if imgContainer already contains an img element
    const existing_img = img_container.querySelector("img");
    if (existing_img) {
      img_container.replaceChild(new_img, existing_img);
    } else {
      img_container.insertBefore(new_img, img_input_label);
    }
  };
};

const custom_form_submit_handler = async (ev) => {
  ev.preventDefault();

  const q_img_input = document.getElementById("q-img");

  //check if there's any input img
  if (!img_input.files || img_input.files.length === 0) {
    const required_content = document.querySelector(".is-required");

    required_content.classList.remove("is-required");
    void required_content.offsetWidth; // force reflow
    required_content.classList.add("is-required");
    required_content.style.display = "block";

    console.warn("Input image is required");
    return;
  }

  q_img_input.files = img_input.files;
  form = ev.target; // don't submit form to server w/.submit()

  // Prepare form data
  const form_data = new FormData(form);
  try {
    const response = await fetch(form.action, {
      method: "POST",
      body: form_data,
    });

    const html_text = await response.text();
    const parsed_doc = new DOMParser().parseFromString(html_text, "text/html");

    // Find the newly rendered selected-imgs div from response
    const new_selected_imgs = parsed_doc.querySelector(".selected-imgs");
    const current_selected_imgs = document.querySelector(".selected-imgs");
    if (new_selected_imgs && current_selected_imgs) {
      current_selected_imgs.innerHTML = new_selected_imgs.innerHTML;
    } else {
      console.warn("Could not update .selected-imgs from response.");
    }

    const n_multiple_faces_form = parsed_doc.querySelector(
      ".multiple-faces-form"
    );
    const c_submit_multiple_faces = document.querySelector(
      ".multiple-faces-form"
    );
    const submit_button = document.querySelector(".submit-multiple-faces");

    if (n_multiple_faces_form && c_submit_multiple_faces) {
      if (n_multiple_faces_form.querySelectorAll("div").length > 0) {
        submit_button.style.display = "block";
        c_submit_multiple_faces.style.display = "block";
      }
      c_submit_multiple_faces.innerHTML = n_multiple_faces_form.innerHTML;
    } else {
      console.warn("Could not find .multiple-input-faces in response.");
    }
  } catch (err) {
    console.error("Failed to submit form:", err);
  }
};

img_input.addEventListener("change", preview_img_handler);
form_element.addEventListener("submit", custom_form_submit_handler);
