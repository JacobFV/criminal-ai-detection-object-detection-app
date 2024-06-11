import React, { useState, useEffect } from "react";
import { detect } from "./core/detect";
import { Class } from "./core/types";

const App = () => {
  const [detectedObjects, setDetectedObjects] = useState([]);
  const [imageElement, setImageElement] = useState<HTMLImageElement | null>(
    null
  );

  useEffect(() => {
    const loadAndDetect = async () => {
      if (imageElement) {
        const classes = [
          new Class("car", [imageElement]), // Add actual images
          new Class("person", [imageElement]), // Add actual images
        ];

        const detected = await detect(imageElement, classes);
        setDetectedObjects(detected);
      }
    };

    loadAndDetect();
  }, [imageElement]);

  return (
    <div>
      <input
        type="file"
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            const img = new Image();
            img.src = URL.createObjectURL(e.target.files[0]);
            img.onload = () => {
              setImageElement(img);
            };
          }
        }}
      />
      <div>
        {detectedObjects.map((obj, index) => (
          <div key={index}>
            <p>
              Detected {obj.classObj.name} with similarity {obj.similarity}
            </p>
            <div>{obj.mask}</div> {/* Ensure mask is displayable */}
          </div>
        ))}
      </div>
    </div>
  );
};

export default App;
