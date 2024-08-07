# %% import libs
import pandas as pd
import numpy as np
import re
import io
import os
    

def read_gmt_lines(gmt_file, comment='#', column_names=['X', 'Y'], encoding='utf-8', read_z=False):
    '''
    Input    : 
        gmt_file   : GMT line segments file
    Output   :
        line_list   : A list for line list
    Added by kfhe, at 01/08/2023
    '''

    # Check if the file exists
    if not os.path.isfile(gmt_file):
        raise FileNotFoundError(f"The file {gmt_file} does not exist.")

    # Read the file content
    with open(gmt_file, 'rt', encoding=encoding) as file:
        file_content = file.read()

    # Remove comment content
    comment_pattern = f"(?<!.)({comment}.*(\\r\\n|\\n)){{1,}}?"
    file_content, _ = re.subn(comment_pattern, '', file_content)

    # Split the content into different segments
    segment_pattern = '>.*?(?:\\r\\n|\\n)'
    segments = re.split(segment_pattern, file_content, flags=re.S)

    # Extract the Z values if required
    if read_z:
        z_values_pattern = '>.*?-Z ?([-0-9.e+]+)'
        z_values = np.array([float(value) for value in re.findall(z_values_pattern, file_content, flags=re.S)])

    # Remove empty segments
    segments = [segment for segment in segments if segment]

    # Convert each segment into a DataFrame
    dataframes = []
    for segment in segments:
        dataframe = pd.read_csv(io.StringIO(segment), sep=r'\s+', names=column_names)
        dataframes.append(dataframe)

    # Remove empty DataFrames
    dataframes = [df for df in dataframes if not df.empty]

    # Return the result
    if read_z:
        return dataframes, z_values
    else:
        return dataframes


def write_lines_to_gmt(segments, z_values, gmt_file=None, csimode=False, coord_precision=3, z_precision=1):
    '''
    Args:
        segments: List of pandas.DataFrame
        z_values: List of z_values

    Kwargs:
        gmt_file: outfile in gmt format; if None print to Screen.
        csimode: Keep.
        coord_precision: Precision of coordinate
        z_precision: Precision of z_value
    '''

    # Check if the number of segments and z_values match
    if len(segments) != len(z_values):
        raise ValueError("The number of segments and z_values must match.")

    # Prepare the format strings
    coord_format = ' '.join([f"{{{{0}}:.{coord_precision}f}}" for _ in range(segments[0].shape[1])])
    z_format = f"> -Z{{0:.{z_precision}f}}"

    # Open the output file
    if gmt_file:
        output_file = open(gmt_file, 'wt')
    else:
        output_file = None

    # Write the segments and z_values to the output file
    try:
        for segment, z_value in zip(segments, z_values):
            print(z_format.format(z_value), file=output_file)
            segment_values = segment.values if segment.__class__ not in (np.ndarray,) else segment
            for point in segment_values:
                print(coord_format.format(*point), file=output_file)
    finally:
        # Close the output file
        if gmt_file:
            output_file.close()

    # All Done
    return


def ReadGMTLines(gmtfile, comment='#', names=['X', 'Y'], encoding='utf-8', readZ=False):
    '''
    Input    : 
        linefile   : GMT line segments file
    Output   :
        lineList   : A list for line list
    Added by kfhe, at 01/08/2023
    '''

    with open(gmtfile, 'rt', encoding=encoding) as fin:
        linestr = fin.read()

    # remove comment content
    comment = '#'
    line_trap, count = re.subn('(?<!.)(' + comment + '.*(\\r\\n|\\n)){1,}?', r'', linestr)
    # Split > to different string
    line_segs = re.split('>.*?(?:\\r\\n|\\n)', line_trap, flags=re.S)
    # Extract value following the Mark Z
    if readZ:
        zvals = np.array([float(izval) for izval in re.findall('>.*?-Z ?([-0-9.e+]+)', line_trap, flags=re.S)])
    # remove null string
    line_segs = [seg for seg in line_segs if seg]
    
    # line segments
    segments = []
    for seg in line_segs:
        segments.append(pd.read_csv(io.StringIO(seg), sep=r'\s+', names=names))
    # Remove empty DataFrame
    if segments[0].empty:
        segments = segments[1:]
    
    # All Done
    if readZ:
        return segments, zvals
    else:
        return segments


def WriteLines2GMT(segs, zval, gmtfile=None, csimode=False, coordtrunc=3, ztrunc=1):
    '''
    Args   :
        * segs       : List of pandas.DataFrame
        * zval       : List of zvals
    
    Kwargs :
        * gmtfile    : outfile in gmt format; if None print to Screen.
        * csimode    : Keep.
        * coordtrunc : Precision of coordinate
        * ztrunc     : Precision of zval
    '''
    ndim = segs[0].shape[1]
    coordpat = ''
    if gmtfile:
        fout = open(gmtfile, 'wt')
    else:
        fout = None
    for i in range(ndim):
        coordpat += '{' + f'{i:d}' + ':.' + f'{coordtrunc:d}'+ 'f} '
    # coordpat = '{0:.3f} {1:.3f} {2:.3f}'
    zpat = '> -Z{0:.' + f'{ztrunc:d}'+ 'f}'
    for iseg, iz in zip(segs, zval):
        print(zpat.format(iz), file=fout)
        iseg = iseg if iseg.__class__ in (np.ndarray,) else iseg.values
        for ipnt in iseg:
            print(coordpat.format(*ipnt), file=fout)
    
    if gmtfile:
        fout.close()
    
    # All Done
    return


if __name__ == '__main__':
    import os
    dirname = r'tests'
    faultfile = r'readgmtlines_test.txt' # 'Haiyuan_Relative_fault.dat'
    linefile = os.path.join(dirname, faultfile)

    segments = ReadGMTLines(linefile)
# %%