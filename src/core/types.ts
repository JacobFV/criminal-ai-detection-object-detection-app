export type Image = HTMLImageElement;

export class Class {
  name: string;
  images: Image[];

  constructor(name: string, images: Image[]) {
    this.name = name;
    this.images = images;
  }
}

export class DetectedObject {
  classObj: Class;
  mask: Image;
  similarity: number;

  constructor(classObj: Class, mask: Image, similarity: number) {
    this.classObj = classObj;
    this.mask = mask;
    this.similarity = similarity;
  }

  get meanLocation(): [number, number] {
    // Implement method to compute mean location of the mask
    return [0, 0]; // Dummy implementation
  }

  get similarityScore(): number {
    // Implement method to compute similarity score
    return this.similarity;
  }
}
