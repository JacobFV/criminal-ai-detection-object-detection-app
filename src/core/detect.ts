// src/detect.ts

import * as tf from "@tensorflow/tfjs";
import { Class, DetectedObject, Image } from "./types";

// Constants (you may need to adjust these)
const THRESHOLD_1 = 0.8; // Similarity threshold for considering a pixel part of the object
const THRESHOLD_2 = 15.0; // Standard deviation threshold for object consistency

async function loadModel(): Promise<tf.GraphModel> {
  return await tf.loadGraphModel("/model/model.json"); // Update with the correct path
}

async function getEmbeddings(
  model: tf.GraphModel,
  image: Image
): Promise<tf.Tensor> {
  const tensor = tf.browser
    .fromPixels(image)
    .expandDims(0)
    .toFloat()
    .div(tf.scalar(127))
    .sub(tf.scalar(1));
  const embeddings = model.execute(tensor) as tf.Tensor;
  return embeddings;
}

function computeSimilarity(
  embedding1: tf.Tensor,
  embedding2: tf.Tensor
): number {
  return tf.losses.cosineDistance(embedding1, embedding2, 0).dataSync()[0];
}

function averagePosition(pixels: [number, number][]): [number, number] {
  if (pixels.length === 0) return [0, 0];
  const xCoords = pixels.map(([x]) => x);
  const yCoords = pixels.map(([, y]) => y);
  return [
    xCoords.reduce((a, b) => a + b, 0) / pixels.length,
    yCoords.reduce((a, b) => a + b, 0) / pixels.length,
  ];
}

function stddev(mean: [number, number], xs: [number, number][]): number {
  if (xs.length === 0) return 0.0;
  const [meanX, meanY] = mean;
  const diffSquareSum = xs.reduce(
    (sum, [x, y]) => sum + (x - meanX) ** 2 + (y - meanY) ** 2,
    0
  );
  return Math.sqrt(diffSquareSum / xs.length);
}

export async function detect(
  image: Image,
  classes: Class[]
): Promise<DetectedObject[]> {
  const model = await loadModel();
  const detectedObjects: DetectedObject[] = [];

  const imageEmbeddings = await getEmbeddings(model, image);

  for (const classObj of classes) {
    let classCentroid: [number, number] | null = null;
    let classStdDev = Infinity;
    let pixelPositions: [number, number][] = []; // Define outside to be accessible later
    let similarities: number[] = []; // Define outside to be accessible later

    for (const classImage of classObj.images) {
      const classEmbeddings = await getEmbeddings(model, classImage);

      pixelPositions = []; // Reset for each class image
      similarities = []; // Reset for each class image

      for (let i = 0; i < image.width; i++) {
        for (let j = 0; j < image.height; j++) {
          const embedding1 = imageEmbeddings.slice(
            [0, i, j, 0],
            [1, 1, 1, imageEmbeddings.shape[3]!]
          );
          const embedding2 = classEmbeddings.slice(
            [0, i, j, 0],
            [1, 1, 1, classEmbeddings.shape[3]!]
          );
          const similarity = computeSimilarity(embedding1, embedding2);
          if (similarity > THRESHOLD_1) {
            similarities.push(similarity);
            pixelPositions.push([i, j]);
          }
        }
      }

      if (pixelPositions.length === 0) continue;

      const centroid = averagePosition(pixelPositions);
      const stdDev =
        stddev(centroid, pixelPositions) / Math.sqrt(pixelPositions.length);

      if (stdDev < THRESHOLD_2) {
        if (classCentroid === null || stdDev < classStdDev) {
          classCentroid = centroid;
          classStdDev = stdDev;
        }
      }
    }

    if (classCentroid !== null) {
      const mask = new Image();
      // Create a blank mask
      const maskCanvas = document.createElement("canvas");
      maskCanvas.width = image.width;
      maskCanvas.height = image.height;
      const ctx = maskCanvas.getContext("2d");
      if (ctx) {
        const imageData = ctx.createImageData(image.width, image.height);
        for (const [x, y] of pixelPositions) {
          const index = (y * image.width + x) * 4;
          imageData.data[index] = 255;
          imageData.data[index + 1] = 255;
          imageData.data[index + 2] = 255;
          imageData.data[index + 3] = 255;
        }
        ctx.putImageData(imageData, 0, 0);
      }
      mask.src = maskCanvas.toDataURL();

      const similarity = Math.max(...similarities);
      detectedObjects.push(new DetectedObject(classObj, mask, similarity));
    }
  }

  return detectedObjects;
}
