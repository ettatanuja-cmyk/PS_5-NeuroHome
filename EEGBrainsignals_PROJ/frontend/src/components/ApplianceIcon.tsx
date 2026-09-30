import { AirVent, Bluetooth, DoorClosed, Lightbulb, Tv, Brain } from 'lucide-react';

export function ApplianceIcon({ name, size = 40 }: { name: string; size?: number }) {
  const cls = 'shrink-0';
  switch (name) {
    case 'Air Conditioner':
      return <AirVent size={size} className={cls} />;
    case 'Bluetooth Speaker':
      return <Bluetooth size={size} className={cls} />;
    case 'Door Lock':
      return <DoorClosed size={size} className={cls} />;
    case 'Electric Light':
      return <Lightbulb size={size} className={cls} />;
    case 'TV':
      return <Tv size={size} className={cls} />;
    default:
      return <Brain size={size} className={cls} />;
  }
}
