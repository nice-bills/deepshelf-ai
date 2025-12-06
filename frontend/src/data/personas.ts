export interface Persona {
  id: string;
  name: string;
  emoji: string;
  description: string;
  history: string[];
}

export const DEMO_PERSONAS: Persona[] = [
  {
    id: 'scifi_fan',
    name: 'The Futurist',
    emoji: '🤖',
    description: 'Loves cyberpunk, AI, and space opera.',
    history: [
      "Neuromancer",
      "Snow Crash",
      "Dune",
      "I, Robot",
      "Do Androids Dream of Electric Sheep?",
      "Foundation"
    ]
  },
  {
    id: 'horror_fan',
    name: 'The Thrill Seeker',
    emoji: '👻',
    description: 'Obsessed with ghosts, psychological terror, and Stephen King.',
    history: [
      "It",
      "The Shining",
      "Dracula",
      "Pet Sematary",
      "The Haunting of Hill House",
      "Misery"
    ]
  },
  {
    id: 'romance_fan',
    name: 'The Hopeless Romantic',
    emoji: '💘',
    description: 'Enjoys period dramas, deep emotions, and happy endings.',
    history: [
      "Pride and Prejudice",
      "Jane Eyre",
      "The Notebook",
      "Sense and Sensibility",
      "Me Before You",
      "Outlander"
    ]
  },
  {
      id: 'history_buff',
      name: 'The Time Traveler',
      emoji: '📜',
      description: 'Fascinated by WWII, ancient civilizations, and biographies.',
      history: [
          "The Diary of a Young Girl",
          "The Guns of August",
          "1776",
          "Sapiens: A Brief History of Humankind",
          "Alexander Hamilton"
      ]
  }
];
