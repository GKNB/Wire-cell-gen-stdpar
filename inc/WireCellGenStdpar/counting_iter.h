#ifndef _STDPAR_COUNTING_ITERATOR_H_
#define _STDPAR_COUNTING_ITERATOR_H_

#include <type_traits>
#include <iterator>

namespace WireCell
{
  namespace GenStdpar
  {
    class counting_iterator
    {
      public:
        typedef size_t          value_type;
        typedef const size_t&   reference;
        typedef const size_t*   pointer;
        typedef typename std::make_signed<size_t>::type   difference_type;
        typedef std::random_access_iterator_tag           iterator_category;
    
        counting_iterator(): m_idx(0) {}
        explicit counting_iterator(size_t x): m_idx(x) {}
    
        value_type operator *() const { return m_idx; }
        
        counting_iterator& operator ++() { ++m_idx; return *this; }
        counting_iterator operator ++(int) { counting_iterator copy{m_idx}; ++m_idx; return copy; }
    
        counting_iterator& operator --() { --m_idx; return *this; }
        counting_iterator operator --(int) { counting_iterator copy(m_idx); --m_idx; return copy; }
        
        counting_iterator& operator +=(difference_type by) { m_idx+=by; return *this; }
        counting_iterator& operator -=(difference_type by) { m_idx-=by; return *this; }
    
        value_type operator [](const difference_type &i) const { return m_idx + i; }
    
        friend counting_iterator operator +(counting_iterator const& i, difference_type n) { return counting_iterator(i.m_idx + n); }
        friend counting_iterator operator +(difference_type n, counting_iterator const& i) { return counting_iterator(i.m_idx + n); }
        friend difference_type   operator -(counting_iterator const& x, counting_iterator const& y) { return x.m_idx - y.m_idx; }
        friend counting_iterator operator -(counting_iterator const& i, difference_type n) { return counting_iterator(i.m_idx - n); }
      
        friend bool operator ==(counting_iterator const& lhs, counting_iterator const& rhs) { return lhs.m_idx == rhs.m_idx; }
        friend bool operator !=(counting_iterator const& lhs, counting_iterator const& rhs) { return !(lhs == rhs); }
    
        friend bool operator < (counting_iterator const& lhs, counting_iterator const& rhs) { return lhs.m_idx < rhs.m_idx; }
        friend bool operator > (counting_iterator const& lhs, counting_iterator const& rhs) { return rhs < lhs; }
        friend bool operator <=(counting_iterator const& lhs, counting_iterator const& rhs) { return !(lhs > rhs); }
        friend bool operator >=(counting_iterator const& lhs, counting_iterator const& rhs) { return !(lhs < rhs); }
    
      private:
        size_t m_idx; // exposition
    };
  } // namespace GenStdpar
} // namespace WireCell

#endif // _STDPAR_COUNTING_ITERATOR_H_
