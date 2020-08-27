# -*- coding: utf-8 -*-

import logging

import numpy as np
import cv2

# User import
from object_detection.configs import cfg as gcfg
from object_detection.configs import ColorPrint


class VizImageGrid:
    """
       (0, 0) ------------------------------------> col
              |
              |     [0, 0]    [0, 1]   ...   [0, n]
              |
              |     [1, 0]    [1, 1]   ...   [1, n]
              |
              |      ...       ...            ...
              |
              |     [n, 0]    [n, 1]   ...   [n, n]
         row  |
    """

    def __init__(self, cell_row, cell_col, grid_row, grid_col):
        logging.warning('__init__( {}, {}, {}, {} )'.format(
            cell_row, cell_col, grid_row, grid_col))

        self.cell_row = cell_row  # 480
        self.cell_col = cell_col  # 640

        self.grid_row = grid_row  # 2
        self.grid_col = grid_col  # 4

        self.gap_sz = 50
        self.border_sz = 25
        self.text_offset = 30

        self.canvas_row = (self.cell_row * self.grid_row
                           + self.gap_sz * (self.grid_row - 1)
                           + self.border_sz * 2)
        self.canvas_col = (self.cell_col * self.grid_col
                           + self.gap_sz * (self.grid_col - 1)
                           + self.border_sz * 2)

        self.color_bg = 128

        self.canvas = np.full((self.canvas_row, self.canvas_col, 3),
                              self.color_bg, np.uint8)

        # text setting
        '''
        https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_HersheyFonts.html
        enum HersheyFonts
        {
            FONT_HERSHEY_SIMPLEX        = 0,
            FONT_HERSHEY_PLAIN          = 1,
            FONT_HERSHEY_DUPLEX         = 2,
            FONT_HERSHEY_COMPLEX        = 3,
            FONT_HERSHEY_TRIPLEX        = 4,
            FONT_HERSHEY_COMPLEX_SMALL  = 5,
            FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
            FONT_HERSHEY_SCRIPT_COMPLEX = 7,
            FONT_ITALIC                 = 16,
        };
        '''
        self.font_face_caption = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale_caption = 1.2
        self.thickness_caption = 3
        self.color_caption = (0, 0, 255)
        self.font_face_subcaption = cv2.FONT_HERSHEY_COMPLEX
        self.font_scale_subcaption = 0.8
        self.thickness_subcaption = 2
        self.color_subcaption = (0, 255, 255)
        self.color_subsubcaption = (255, 0, 255)

        if not gcfg.get_is_release and gcfg.get_is_log_enabled:
            ColorPrint.print_warn(('  --> VizImageGrid::'
                                   '__init__( {}, {}, {}, {} )').format(
                cell_row, cell_col, grid_row, grid_col))
            ColorPrint.print_warn('  --> canvas:                 {}x{}'.format(
                self.canvas_row, self.canvas_col))

    def _check_input(self, row, col, data):
        if not isinstance(row, int) or not isinstance(col, int)\
                or not isinstance(data, np.ndarray):
            raise ValueError
        if row < 0 or row >= self.grid_row:
            raise ValueError
        if col < 0 or col >= self.grid_col:
            raise ValueError
        data_shape = data.shape
        if len(data_shape) != 3:
            raise ValueError
        if data_shape[0] != self.cell_row or data_shape[1] != self.cell_col\
                or data_shape[2] != 3:
            raise ValueError

    def caption(self, caption_text):
        logging.warning('caption( {} )'.format(caption_text))

        # compute offset
        '''
        https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#gettextsize
            cv2.getTextSize(text, fontFace, fontScale, thickness)
              → retval, baseLine
        '''
        retval, base_line = cv2.getTextSize(
            caption_text, self.font_face_caption, self.font_scale_caption,
            self.thickness_caption)
        box_col, box_row = retval

        # offset X
        pos_col = self.text_offset + self.border_sz
        if box_col < self.canvas_col:
            pos_col = int((self.canvas_col - box_col) / 2)

        # offset Y
        pos_row = self.border_sz + box_row + self.thickness_caption
        if box_row < self.gap_sz:
            pos_row = int(self.border_sz + (self.gap_sz - box_row) / 2 + box_row)

        if self.grid_row >= 2:
            pos_row += self.cell_row

        cv2.putText(self.canvas, caption_text, (pos_col, pos_row),
                    self.font_face_caption, self.font_scale_caption,
                    self.color_caption, self.thickness_caption)

    def subplot(self, row, col, data, caption_text):
        """
        Add a subplot to canvas.

        Parameters
        ----------
        row : int
        col : int
        data : numpy.ndarray
        caption_text : str
        """
        logging.warning('subplot( {}, {}, {} )'.format(row, col, data.shape))

        self._check_input(row, col, data)
        row_beg = (self.cell_row + self.gap_sz) * row + self.border_sz
        row_end = row_beg + self.cell_row
        col_beg = (self.cell_col + self.gap_sz) * col + self.border_sz
        col_end = col_beg + self.cell_col
        self.canvas[row_beg:row_end, col_beg:col_end, :] = data

        if caption_text != '':
            '''
            cv2.putText(img, text, org, fontFace, fontScale, color,
                        thickness=None, lineType=None, bottomLeftOrigin=None)
            '''
            retval, base_line = cv2.getTextSize(
                'TEST', self.font_face_subcaption, self.font_scale_subcaption,
                self.thickness_subcaption)
            box_col, box_row = retval

            for i, str_line in enumerate(caption_text.split('\n')):
                if i == 0:
                    color = self.color_subcaption
                else:
                    color = self.color_subsubcaption
                pos_col = col_beg + self.text_offset
                pos_row = row_beg + self.text_offset + i * box_row * 2
                cv2.putText(self.canvas, str_line, (pos_col, pos_row),
                            self.font_face_subcaption,
                            self.font_scale_subcaption,
                            color, self.thickness_subcaption)

    def save_fig(self, path):
        logging.warning('save_fig( {} )'.format(path))
        cv2.imwrite(path, self.canvas)

    def create_palette(self, title, names, colors, use_rect):
        """
        Visualize the palette.

        Parameters
        ----------
        title : str
        names : list of str
        colors : numpy.ndarray
        use_rect : bool
        """
        logging.warning('create_palette( {}, {}, {} )'.format(
            title, names, colors.shape))
        if colors.shape[0] != len(list(names)):
            raise ValueError

        font_title = cv2.FONT_HERSHEY_SIMPLEX
        # get text size
        retval_title, base_line_title = cv2.getTextSize(
            title, font_title, self.font_scale_caption,
            self.thickness_caption)
        _, box_row_title = retval_title

        retval_text, base_line_text = cv2.getTextSize(
            title, self.font_face_subcaption, self.font_scale_subcaption,
            self.thickness_subcaption)
        _, box_row_text = retval_text

        # create image and draw title text
        img_palette = np.zeros((self.cell_row, self.cell_col, 3), np.uint8)
        pos_col = self.text_offset
        pos_row = self.text_offset + box_row_title

        cv2.putText(img_palette, title, (pos_col, pos_row),
                    font_title, self.font_scale_caption,
                    self.color_caption, self.thickness_caption)

        # plot legend
        square_gap = 15
        if use_rect:
            square_len_col = 100
            square_len_row = 5
        else:
            square_len_col = 25
            square_len_row = 25

        pos_square_col_beg = self.text_offset
        pos_square_row_beg = pos_row + box_row_title + square_gap * 2
        pos_text_col_beg = pos_square_col_beg + square_len_col + square_gap * 2
        if use_rect:
            pos_text_row_beg = int(pos_square_row_beg + box_row_text / 2)
        else:
            pos_text_row_beg = pos_square_row_beg + box_row_text
        pos_space = max(square_len_row, box_row_text) + square_gap

        n_max_row = int((self.cell_row - pos_square_row_beg) / pos_space)
        n_max_col = colors.shape[0] // n_max_row
        if colors.shape[0] % n_max_row != 0:
            n_max_col += 1
        col_offset = int((self.cell_col - pos_square_row_beg -
                          square_len_row - pos_space) / n_max_col)

        for idx in range(colors.shape[0]):
            square_color = colors[idx]
            square_name = names[idx]

            idx_col = idx / n_max_row
            idx_row = idx % n_max_row

            # square or rectangle
            pos_square_col_min = int(pos_square_col_beg + idx_col * col_offset)
            pos_square_col_max = int(pos_square_col_min + square_len_col)
            pos_square_row_min = int(pos_square_row_beg + pos_space * idx_row)
            pos_square_row_max = int(pos_square_row_min + square_len_row)

            img_palette[pos_square_row_min:pos_square_row_max,
                        pos_square_col_min:pos_square_col_max] = square_color

            # text
            pos_text_col_min = int(pos_text_col_beg + idx_col * col_offset)
            pos_text_row_min = int(pos_text_row_beg + pos_space * idx_row)
            cv2.putText(img_palette, square_name,
                        (pos_text_col_min, pos_text_row_min),
                        self.font_face_subcaption, self.font_scale_subcaption,
                        square_color, self.thickness_subcaption)
        return img_palette

    def add_lane_matching(self, img, matching, color_key, color_value):
        logging.warning('add_lane_matching( .. )')

        font_title = cv2.FONT_HERSHEY_SIMPLEX

        title = 'Title'

        # get text size
        retval_title, _ = cv2.getTextSize(
            title, font_title, self.font_scale_caption,
            self.thickness_caption)
        _, box_row_title = retval_title

        retval_text, _ = cv2.getTextSize(
            title, self.font_face_subcaption, self.font_scale_subcaption,
            self.thickness_subcaption)
        _, box_row_text = retval_text

        # create image and draw title text
        pos_row = self.text_offset + box_row_title

        # plot legend
        square_gap = 15
        pos_text_col_beg = self.text_offset
        pos_text_row_beg = pos_row + box_row_title + square_gap * 2
        pos_space = box_row_text + square_gap

        n_map = len(list(matching.keys()))
        key_arr = list(matching.keys())
        key_arr.sort()
        for idx in range(n_map):
            idx_row = idx

            # text - key
            pos_text_col_min = int(pos_text_col_beg)
            pos_text_row_min = int(pos_text_row_beg + pos_space * idx_row)
            cv2.putText(img, '{} -'.format(key_arr[idx]),
                        (pos_text_col_min, pos_text_row_min),
                        self.font_face_subcaption, self.font_scale_subcaption,
                        color_key, self.thickness_subcaption)

            # text - value
            retval_, _ = cv2.getTextSize(
                '{} -'.format(key_arr[idx]),
                self.font_face_subcaption, self.font_scale_subcaption,
                self.thickness_subcaption)
            box_col, _ = retval_
            cv2.putText(img, '-> {}'.format(matching[key_arr[idx]]),
                        (pos_text_col_min + box_col, pos_text_row_min),
                        self.font_face_subcaption, self.font_scale_subcaption,
                        color_value, self.thickness_subcaption)
