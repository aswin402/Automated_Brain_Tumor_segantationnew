export function Button({ 
  className = "", 
  variant = "primary", 
  size = "md",
  children, 
  disabled,
  ...props 
}) {
  const baseStyles = "font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2"
  
  const variants = {
    primary: "bg-blue-600 hover:bg-blue-700 text-white disabled:bg-slate-600",
    secondary: "bg-slate-700 hover:bg-slate-600 text-white disabled:bg-slate-600",
    outline: "border border-slate-600 hover:bg-slate-700 text-white"
  }

  const sizes = {
    sm: "px-3 py-1 text-sm",
    md: "px-4 py-2 text-base",
    lg: "px-6 py-3 text-lg"
  }

  return (
    <button 
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  )
}
