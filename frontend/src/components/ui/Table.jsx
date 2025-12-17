export function Table({ className = "", children, ...props }) {
  return (
    <div className="overflow-x-auto">
      <table className={`w-full text-sm ${className}`} {...props}>
        {children}
      </table>
    </div>
  )
}

export function TableHeader({ className = "", children, ...props }) {
  return (
    <thead className={`bg-slate-700 border-b border-slate-600 ${className}`} {...props}>
      {children}
    </thead>
  )
}

export function TableBody({ className = "", children, ...props }) {
  return (
    <tbody className={className} {...props}>
      {children}
    </tbody>
  )
}

export function TableRow({ className = "", children, ...props }) {
  return (
    <tr className={`border-b border-slate-700 hover:bg-slate-700/50 transition-colors ${className}`} {...props}>
      {children}
    </tr>
  )
}

export function TableHead({ className = "", children, ...props }) {
  return (
    <th className={`px-4 py-3 text-left font-semibold text-white ${className}`} {...props}>
      {children}
    </th>
  )
}

export function TableCell({ className = "", children, ...props }) {
  return (
    <td className={`px-4 py-3 text-slate-300 ${className}`} {...props}>
      {children}
    </td>
  )
}
