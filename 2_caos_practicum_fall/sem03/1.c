STYPE 
bit_reverse(STYPE value) 
{
    UTYPE for_check = value | ~value;
    UTYPE new_value = 0;
    while (for_check > 0) {
        new_value <<= 1;
        new_value += value & 1;
        value >>= 1;
        for_check >>= 1;
    }
    return (STYPE) new_value;
}


