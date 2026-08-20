import AppKit
import Foundation
import Vision

guard CommandLine.arguments.count == 3 else {
    FileHandle.standardError.write(Data("usage: vision_ocr.swift IMAGE LANGUAGE\n".utf8))
    exit(2)
}
let path = CommandLine.arguments[1]
let language = CommandLine.arguments[2]
guard let image = NSImage(contentsOfFile: path),
      let data = image.tiffRepresentation,
      let bitmap = NSBitmapImageRep(data: data),
      let cgImage = bitmap.cgImage else {
    FileHandle.standardError.write(Data("cannot decode image\n".utf8))
    exit(3)
}
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.recognitionLanguages = [language]
request.usesLanguageCorrection = false
let handler = VNImageRequestHandler(cgImage: cgImage)
do {
    try handler.perform([request])
    let lines = (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }
    print(lines.joined(separator: "\n"))
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(4)
}
