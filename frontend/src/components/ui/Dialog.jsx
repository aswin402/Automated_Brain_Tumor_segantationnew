import { useState } from 'react'

export function Dialog({ open, onOpenChange, children }) {
  return (
    <div className={`fixed inset-0 z-50 ${open ? 'block' : 'hidden'}`}>
      <div 
        className="fixed inset-0 bg-black/50"
        onClick={() => onOpenChange(false)}
      />
      <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-slate-800 rounded-lg border border-slate-700 p-6 max-w-md w-full z-50">
        {children}
      </div>
    </div>
  )
}

export function DialogContent({ className = "", children, ...props }) {
  return <div className={className} {...props}>{children}</div>
}

export function DialogHeader({ className = "", children, ...props }) {
  return <div className={`mb-4 ${className}`} {...props}>{children}</div>
}

export function DialogTitle({ className = "", children, ...props }) {
  return <h2 className={`text-lg font-bold text-white ${className}`} {...props}>{children}</h2>
}

export function DialogFooter({ className = "", children, ...props }) {
  return <div className={`mt-6 flex justify-end gap-2 ${className}`} {...props}>{children}</div>
}
