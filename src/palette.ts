const WATER_DEEP_TO_FOAM = ['#141f33', '#1f3560', '#2a527c', '#3a6886', '#4a7c8f', '#5a8c99', '#7aa0a7', '#a9c2bd', '#e8f5f2'];
const DUCK_OUTLINE_TO_HIGHLIGHT = ['#241e0e', '#5e4c0e', '#8e7722', '#bf9a14', '#e0b92a', '#f6d43a', '#fbe983', '#fdf6c4'];
const BEAK = ['#6b3310', '#b0561a', '#f57a1c'];
const ROCK = ['#1f1d36', '#3f4577', '#5c6189', '#7d849e', '#a6b3c2'];
const MOSS_AND_LILY = ['#2a3412', '#46603a', '#6c9a48', '#93bd62', '#c4dd94'];
const FLOWER = ['#a0506f', '#f59ac6'];

export const PALETTE: string[] = [...WATER_DEEP_TO_FOAM, ...DUCK_OUTLINE_TO_HIGHLIGHT, ...BEAK, ...ROCK, ...MOSS_AND_LILY, ...FLOWER];

export function rgb(): [number, number, number][] {
    return PALETTE.map((hex) => {
        const n = Number.parseInt(hex.slice(1), 16);
        return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
    });
}
