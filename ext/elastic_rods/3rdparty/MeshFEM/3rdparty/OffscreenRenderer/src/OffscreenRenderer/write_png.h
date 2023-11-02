#ifndef WRITE_PNG_H
#define WRITE_PNG_H

#include <png.h>

// Extremely basic wrapper for libpng--write 8 bit RGBA image with no error checking.
inline void write_png_RGBA(const std::string &path, int width, int height, const unsigned char *data, bool verticalFlip = false) {
        FILE *fp = fopen(path.c_str(), "wb");
        if (!fp) throw std::runtime_error("Could not open " + path);

        auto writeStruct = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        png_init_io(writeStruct, fp);

        auto info = png_create_info_struct(writeStruct);
        png_set_IHDR(writeStruct, info, width, height,
                     8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

        png_write_info(writeStruct, info);

        std::vector<png_bytep> rowPointers(height);
        for (int row = 0; row < height; ++row) {
            int i = verticalFlip ? (height - 1) - row : row;
            rowPointers[row] = png_bytep(data + i * (width * 4));
        }

        png_write_image(writeStruct, rowPointers.data());
        png_write_end(writeStruct, NULL);

        fclose(fp);
}

#endif /* end of include guard: WRITE_PNG_H */
