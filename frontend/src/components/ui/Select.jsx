export function Select({ className = "", children, ...props }) {
  return (
    <select 
      className={`w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 ${className}`}
      {...props}
    >
      {children}
    </select>
  )
}

export function SelectOption({ value, children }) {
  return <option value={value}>{children}</option>
}
